"""The dry end-to-end run's virtual clock inside the lab jobs (e2e_dry.py puts this folder on PYTHONPATH with
E2E_CLOCK_OFFSET = virtual now − real now, in seconds): time.time and datetime.now / utcnow move with the
workflow's sys.now(), so a job's timestamps (the drift debounce, lock expiries, trained_at) agree with the
workflow's. Test-only: never on a real job's path."""
import os

_off = float(os.environ.get("E2E_CLOCK_OFFSET") or 0)
if _off:
    import datetime as _dt
    import time as _time

    _real_time = _time.time
    _time.time = lambda: _real_time() + _off
    _real_time_ns = _time.time_ns
    _time.time_ns = lambda: _real_time_ns() + int(_off * 1e9)

    class _DateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return super().now(tz) + _dt.timedelta(seconds=_off)

        @classmethod
        def utcnow(cls):
            return super().utcnow() + _dt.timedelta(seconds=_off)

    _dt.datetime = _DateTime
