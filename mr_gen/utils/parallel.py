from warnings import warn
from typing import List, Callable, Any
from joblib import Parallel, delayed
from tqdm import tqdm


def parallel_luncher(
    job: Callable, argset: List[Any], pnum: int, unpack=False, use_tq=True, **kwargs
) -> list:
    """Parallel's delayed method luncher."""

    def job_wrraper(job: Callable, unpack: bool = False) -> Callable:
        """Wrapper for job that have dict type args."""
        if unpack:
            if isinstance(argset[0], dict):
                return lambda _args: job(**_args)
            elif isinstance(argset[0], (list, tuple)):
                return lambda _args: job(*_args)
            else:
                warn("Parallel job argments cannot be unpacked.")
        return job

    result = []
    iterator = range(len(argset))
    iterator = tqdm(iterator, **kwargs) if use_tq else iterator

    result = Parallel(n_jobs=pnum, verbose=0)(
        delayed(job_wrraper(job, unpack))(argset[i]) for i in iterator
    )
    return list(result)
