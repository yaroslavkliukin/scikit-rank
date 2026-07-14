from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def reset_accelerate_state() -> None:
    """Keep Accelerate's process-wide singleton from leaking between tests."""

    def reset() -> None:
        try:
            from accelerate.state import AcceleratorState

            AcceleratorState._reset_state(reset_partial_state=True)
        except (ImportError, AttributeError):
            pass

    reset()
    yield
    reset()
