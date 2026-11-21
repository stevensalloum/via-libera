"""Token budget guard for TabPFN cloud calls.

Every cloud call costs at least 10,000 tokens (measured 2026-11-21); large test sets cost
more. ``CostGuard`` asks for a free quote first and refuses calls above a limit.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# quote(train_rows, test_rows, columns) -> estimated tokens
Quote = Callable[[int, int, int], int]


class BudgetExceeded(RuntimeError):
    """A single call would cost more than the guard allows."""


def cloud_quote(train_rows: int, test_rows: int, columns: int) -> int:
    """Free server quote via ``tabpfn_client.estimate_cost`` (sends only shapes)."""
    from tabpfn_client import estimate_cost

    zero = np.float32(0)  # broadcast views: O(1) memory
    train = np.broadcast_to(zero, (train_rows, columns))
    test = np.broadcast_to(zero, (max(test_rows, 1), columns))
    return estimate_cost(train, test).estimated_cost


@dataclass
class CostGuard:
    """Refuse any call quoted above ``max_tokens_per_call``; tally accepted calls."""

    max_tokens_per_call: int = 200_000
    quote: Quote = cloud_quote
    spent: int = 0
    calls: int = 0

    def check(self, train_rows: int, test_rows: int, columns: int) -> int:
        """Quote one call; raise ``BudgetExceeded`` or record it and return its cost."""
        cost = self.quote(train_rows, test_rows, columns)
        if cost > self.max_tokens_per_call:
            raise BudgetExceeded(
                f"call would cost {cost:,} tokens (limit {self.max_tokens_per_call:,}); "
                "subsample the test rows or raise the limit deliberately"
            )
        self.spent += cost
        self.calls += 1
        return cost
