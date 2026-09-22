"""Terminal progress display for ``predict``: one bar per worker, reused for each tomogram."""
from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.progress_bar import ProgressBar
from rich.table import Column
from rich.text import Text

from tomo_slab.runner import ProgressEvent


class _Bar(BarColumn):
    """A bar that pulses while the current phase has no measurable progress."""

    def render(self, task):
        if task.fields.get("busy"):
            return ProgressBar(
                total=None, pulse=True, width=self.bar_width, animation_time=task.get_time()
            )
        return super().render(task)


class _Percent(TaskProgressColumn):
    def render(self, task):
        return Text("") if task.fields.get("busy") else super().render(task)


def _shorten(text: str, width: int) -> str:
    """Cut ``text`` to ``width`` characters by replacing the middle with an ellipsis."""
    if len(text) <= width:
        return text
    tail = (width - 1) // 2
    return text[: width - 1 - tail] + "…" + text[len(text) - tail :]


class ProgressView:
    """One bar per worker; a worker's bar moves on to its next tomogram when it finishes one.

    There are never more bars than workers, so the display keeps the same height for the whole
    run. Every row has a fixed layout that fits the terminal, since Rich mis-erases a live
    display whose lines wrap. Bars go to stderr and are transient; on a non-terminal nothing is
    drawn. Use as a context manager, and print through `print` while it is active.
    """

    _PHASE_WIDTH = len("Slab Blending (XZ axis)")
    _FIXED_WIDTH = 36  # bar, percentage, elapsed time and the gaps between the columns

    def __init__(self, devices: Sequence[str], names: Sequence[str] = ()) -> None:
        """``devices`` has one entry per worker (its bar); ``names`` are the tomogram names."""
        self._console = Console(stderr=True)
        self._progress = Progress(
            TextColumn(
                "{task.description}", table_column=Column(no_wrap=True, overflow="ellipsis")
            ),
            _Bar(bar_width=20),
            _Percent(),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        self._devices = list(devices)
        self._device_width = max(map(len, devices), default=0)
        room = self._console.width - self._FIXED_WIDTH - self._PHASE_WIDTH - 2
        longest = max(map(len, names), default=0)
        self._name_width = max(8, min(longest, room - self._device_width))
        self._lock = threading.Lock()
        self._current: dict[int, int] = {}  # worker -> index of the tomogram it is showing
        self._rows: list[TaskID] = [
            self._progress.add_task(self._text(device, "", "starting up"), total=1, busy=True)
            for device in devices
        ]

    def _text(self, device: str, name: str, phase: str) -> str:
        return (
            f"{escape(device):<{self._device_width}} "
            f"{escape(_shorten(name, self._name_width).ljust(self._name_width))} "
            f"[dim]{escape(phase.ljust(self._PHASE_WIDTH))}[/dim]"
        )

    def handle(self, event: ProgressEvent) -> None:
        """Apply one worker event. Safe to call from another thread."""
        with self._lock:
            if event.kind == "finish":
                self._idle(event.slot, event.index)
                return
            if event.kind == "start":
                self._current[event.slot] = event.index
            elif self._current.get(event.slot) != event.index:
                return  # a late event for a tomogram that is already done
            busy = event.total is None
            update = self._progress.reset if event.kind == "start" else self._progress.update
            update(
                self._rows[event.slot],
                description=self._text(event.device, event.name, event.phase),
                total=1 if busy else event.total,
                completed=0 if busy else event.completed,
                busy=busy,
            )

    def _idle(self, slot: int, index: int) -> None:
        """Blank a worker's bar, unless it has already moved on to another tomogram."""
        if self._current.get(slot) != index:
            return
        del self._current[slot]
        self._progress.reset(
            self._rows[slot], start=False, total=1, completed=0, busy=False,
            description=self._text(self._devices[slot], "idle", ""),
        )

    def complete(self, index: int) -> None:
        """Mark a tomogram as done (whatever its outcome)."""
        with self._lock:
            for slot, current in list(self._current.items()):
                if current == index:
                    self._idle(slot, index)

    def print(self, message: str, style: Optional[str] = None, err: bool = False) -> None:
        """Print a line above the bars (or, without a terminal, to stdout/stderr as usual)."""
        if self._console.is_terminal:
            self._console.print(Text(message, style=style or ""), highlight=False)
        else:
            typer.secho(message, fg=style, err=err)

    def __enter__(self) -> ProgressView:
        self._progress.start()
        return self

    def __exit__(self, *exc) -> None:
        self._progress.stop()
