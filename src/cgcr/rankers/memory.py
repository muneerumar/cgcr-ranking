"""Mac memory telemetry and the explicit 1.5 decimal-GB safety cutoff."""

MIN_AVAILABLE_BYTES=1_500_000_000


class InsufficientMemory(MemoryError):
    pass


def memory_snapshot():
    import os
    import psutil
    process=psutil.Process(os.getpid())
    return {'resident_bytes':process.memory_info().rss,'available_bytes':psutil.virtual_memory().available}


def ensure_available(minimum=MIN_AVAILABLE_BYTES):
    memory=memory_snapshot()
    if memory['available_bytes']<minimum:
        raise InsufficientMemory(f"Available memory {memory['available_bytes']} bytes below required {minimum}")
    return memory
