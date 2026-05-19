from typing import List
from itertools import combinations


def all_subsets_str(elements: List[str]) -> List[str]:
    return [
        ",".join(combo)
        for r in range(1, len(elements) + 1)
        for combo in combinations(elements, r)
    ]