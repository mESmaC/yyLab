from __future__ import annotations

import threading
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    kind: str
    status: str = "running"
    progress: dict = field(default_factory=lambda: {"stage": "starting", "frac": 0.0, "message": ""})
    result: object = None
    error: str = ""


ProgressFn = Callable[[str, float, str], None]


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn, *args, **kwargs) -> str:
        job = Job(id=uuid.uuid4().hex[:8], kind=kind)

        def progress(stage: str = "", frac: float = 0.0, message: str = "") -> None:
            job.progress = {"stage": stage, "frac": max(0.0, min(1.0, frac)), "message": message}

        def target() -> None:
            try:
                job.result = fn(progress, *args, **kwargs)
                job.status = "done"
                progress(job.progress["stage"], 1.0, "finished")
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.error = f"{e}\n{traceback.format_exc()}"

        with self._lock:
            self._jobs[job.id] = job
        threading.Thread(target=target, daemon=True, name=f"yylab-job-{job.id}").start()
        return job.id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())


JOBS = JobManager()
