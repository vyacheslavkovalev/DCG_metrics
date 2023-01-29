from typing import List

import numpy as np


def avg_ndcg(
    list_relevances: List[List[float]], k: int, method: str = "standard"
) -> float:
    """avarage nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if k > len(list_relevances[0]):
        raise IndexError()
    if method == "standard":
        score = np.divide(
            np.array(list_relevances)[:, :k],
            len(list_relevances)
            * [np.log2(np.arange(2, len(list_relevances[0][:k]) + 2))],
        ).sum(1)
        max_score = np.divide(
            np.sort(list_relevances)[:, ::-1][:, :k],
            len(list_relevances)
            * [np.log2(np.arange(2, len(list_relevances[0][:k]) + 2))],
        ).sum(1)
    elif method == "industry":
        score = np.divide(
            (2 ** np.array(list_relevances) - 1)[:, :k],
            len(list_relevances)
            * [np.log2(np.arange(2, len(list_relevances[0][:k]) + 2))],
        )[:, ::-1].sum(1)
        max_score = np.divide(
            (2 ** np.sort(list_relevances)[:, ::-1][:, :k] - 1),
            len(list_relevances)
            * [np.log2(np.arange(2, len(list_relevances[0][:k]) + 2))],
        ).sum(1)
    else:
        raise ValueError()
    return np.average(score / max_score)
