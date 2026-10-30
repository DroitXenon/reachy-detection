"""Run inference on the latest frame without blocking the camera preview."""

from dataclasses import dataclass
from threading import Condition, Thread
import time


@dataclass(frozen=True)
class FrameResult:
    sequence: int
    generation: int
    captured_at: float
    completed_at: float
    inference_ms: float
    value: object


class LatestFrameWorker:
    """One frame in flight and one replaceable pending frame; never a backlog.

    The factory returns a context-managed callable owned only by this thread.
    A generation identifies the current camera orientation/size. Old generations
    and results older than max_age are never drawn over a new frame.
    """

    def __init__(self, factory):
        self._factory = factory
        self._condition = Condition()
        self._pending = None
        self._result = None
        self._error = None
        self._stopped = False
        self._thread = Thread(target=self._run, name="vision-inference", daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def submit(self, frame, sequence, captured_at, generation):
        # Caller owns and annotates its frame; inference must see a clean copy.
        item = (frame.copy(), sequence, captured_at, generation)
        with self._condition:
            if not self._stopped:
                self._pending = item
                self._condition.notify()

    def latest(self, generation, max_age=0.30):
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"Vision worker failed: {self._error}") from self._error
            result = self._result
        if (result is None or result.generation != generation
                or time.monotonic() - result.captured_at > max_age):
            return None
        return result

    def _run(self):
        try:
            with self._factory() as processor:
                while True:
                    with self._condition:
                        self._condition.wait_for(lambda: self._stopped or self._pending is not None)
                        if self._stopped:
                            return
                        frame, sequence, captured_at, generation = self._pending
                        self._pending = None
                    started = time.monotonic()
                    value = processor(frame, sequence)
                    finished = time.monotonic()
                    result = FrameResult(sequence, generation, captured_at, finished,
                                         (finished - started) * 1000, value)
                    with self._condition:
                        self._result = result
        except Exception as error:
            with self._condition:
                self._error = error

    def __exit__(self, *exc):
        with self._condition:
            self._stopped = True
            self._pending = None
            self._condition.notify()
        self._thread.join(timeout=5)
