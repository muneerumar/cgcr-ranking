"""the available-memory threshold is enforced without loading any model."""
import pytest
import cgcr.rankers.memory as memory


def test_available_memory_cutoff(monkeypatch):
    monkeypatch.setattr(memory,'memory_snapshot',lambda:{'resident_bytes':10,'available_bytes':1_499_999_999})
    with pytest.raises(memory.InsufficientMemory,match='below required'):memory.ensure_available()
    monkeypatch.setattr(memory,'memory_snapshot',lambda:{'resident_bytes':10,'available_bytes':1_500_000_000})
    assert memory.ensure_available()['available_bytes']==1_500_000_000
