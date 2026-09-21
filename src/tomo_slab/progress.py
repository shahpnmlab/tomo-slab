"""Terminal progress display for ``predict``: one row per tomogram being processed."""
from __future__ import annotations

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


class ProgressView:
    """An overall bar plus one row for each tomogram currently being processed.

    Bars go to stderr and are transient; on a non-terminal nothing is drawn. Use as a context
    manager; while it is active, ordinary prints are shown above the bars.
    """

    def __init__(self, n_tomograms: int) -> None:
        self._progress = Progress(
            TextColumn("{task.description}"),
            _Bar(bar_width=30),
            _Percent(),
            TimeElapsedColumn(),
            console=Console(stderr=True),
            transient=True,
        )
        self._n = n_tomograms
        self._done = 0
        self._overall = self._progress.add_task(self._overall_text(), total=n_tomograms)
        self._rows: dict[int, TaskID] = {}
        self._finished: set[int] = set()

    def _overall_text(self) -> str:
        return f"[bold]Tomograms {self._done}/{self._n}[/bold]"

    @staticmethod
    def _row_text(event: ProgressEvent) -> str:
        return f"  {event.device:<7} {escape(event.name)} [dim]{event.phase}[/dim]"

    def handle(self, event: ProgressEvent) -> None:
        """Apply one worker event. Safe to call from another thread."""
        if event.index in self._finished:
            return  # a late event for a tomogram that is already done
        if event.kind == "finish":
            self._remove(event.index)
            return
        busy = event.total is None
        fields = {"busy": busy}
        row = self._rows.get(event.index)
        if row is None:
            row = self._rows[event.index] = self._progress.add_task(
                self._row_text(event), total=1, **fields
            )
        self._progress.update(
            row,
            description=self._row_text(event),
            total=1 if busy else event.total,
            completed=0 if busy else event.completed,
            **fields,
        )

    def _remove(self, index: int) -> None:
        self._finished.add(index)
        row = self._rows.pop(index, None)
        if row is not None:
            self._progress.remove_task(row)

    def complete(self, index: int) -> None:
        """Mark a tomogram as done (whatever its outcome)."""
        self._remove(index)
        self._done += 1
        self._progress.update(self._overall, advance=1, description=self._overall_text())

    def __enter__(self) -> ProgressView:
        self._progress.start()
        return self

    def __exit__(self, *exc) -> None:
        self._progress.stop()
