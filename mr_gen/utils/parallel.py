from warnings import warn
from typing import Tuple, List, Callable, Any
from joblib import Parallel, delayed
from tqdm import tqdm


def _delayed_arg_assign(
    job: Callable, arg_list: list, pnum: int, unpack: bool = False
) -> List[Tuple]:
    if unpack:
        if isinstance(arg_list[0], (list, tuple)):
            return [delayed(job)(*arg_list[i]) for i in range(pnum)]
        elif isinstance(arg_list[0], dict):
            return [delayed(job)(**arg_list[i]) for i in range(pnum)]
        else:
            warn("Parallel job argments cannot be unpacked.")
    return [delayed(job)(arg_list[i]) for i in range(pnum)]


def parallel_luncher(
    job: Callable, arg_list: list, pnum: int, unpack: bool = False, pn="", use_tq=True
) -> List[Any]:
    """Parallel's delayed method luncher."""

    result = []
    iterator = range(0, len(arg_list), pnum)
    iterator = tqdm(iterator, pn) if use_tq else iterator

    for step in iterator:
        pnum = min(pnum, len(arg_list) - step)
        result += Parallel(n_jobs=pnum, verbose=0)(
            _delayed_arg_assign(job, arg_list[step : step + pnum], pnum, unpack)
        )
    return result
