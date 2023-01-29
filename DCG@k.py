from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values​​
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if k > len(relevance):
        raise IndexError()

    if method == "standard":
        score = (np.array(relevance) / np.log2(np.arange(2, len(relevance)+2)))[:k].sum()
    elif method == "industry":
        score = ((2**np.array(relevance) - 1) / np.log2(np.arange(2, len(relevance)+2)))[:k].sum()
    else:
        raise ValueError()
    return score
