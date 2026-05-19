# -*- coding: utf-8 -*-
"""
    SCoPE
    Dissimilarity Metrics
    Jesus Alan Hernandez Galvan
"""


def _ncd(c_x1: float, c_x2: float, c_x1x2: float) -> float:
    """Normalized Compression Distance"""

    numerator_: float = c_x1x2 - min(c_x1, c_x2)
    denominator_: float = max(c_x1, c_x2)

    return numerator_ / denominator_


def _cdm(c_x1: float, c_x2: float, c_x1x2: float) -> float:
    """Compression Dissimilarity Measure"""

    numerator_ = c_x1x2
    denominator_ = c_x1 + c_x2

    result = numerator_ / denominator_

    return result


def _clm(c_x1: float, c_x2: float, c_x1x2: float) -> float:

    numerator_ = 1 - (c_x1 + c_x2 - c_x1x2)
    denominator_ = c_x1x2

    result = denominator_ / numerator_

    return result