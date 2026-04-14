import time
from backend.utils.fps_controller import TimeBasedFPSController


class FakeTime:
    """
    Simple controllable clock for testing.
    """
    def __init__(self, start=0.0):
        self.current = start

    def advance(self, seconds):
        self.current += seconds

    def time(self):
        return self.current


def test_fps_controller(monkeypatch):
    print("\n=== TimeBasedFPSController VERBOSE TEST ===")

    # Create fake clock starting at T=1000
    fake_clock = FakeTime(start=1000.0)
    monkeypatch.setattr(time, "time", fake_clock.time)

    # Target 1 FPS → 1 second interval
    fps = TimeBasedFPSController(target_fps=1)

    print(f"Target FPS: 1")
    print(f"Interval: {fps.interval} sec\n")

    # ---- First call ----
    print("[T=1000.0] First should_process() call")
    result = fps.should_process()
    print("Result:", result)
    assert result is True

    # ---- Advance 0.3 sec ----
    fake_clock.advance(0.3)
    print("\n[T=1000.3] Second should_process() call")
    result = fps.should_process()
    print("Result:", result)
    assert result is False

    # ---- Advance to 1.1 sec total ----
    fake_clock.advance(0.8)
    print("\n[T=1001.1] Third should_process() call")
    result = fps.should_process()
    print("Result:", result)
    assert result is True

    # ---- Immediate call again ----
    print("\n[T=1001.1] Fourth should_process() call")
    result = fps.should_process()
    print("Result:", result)
    assert result is False

    print("\n=== TEST COMPLETED SUCCESSFULLY ===\n")
