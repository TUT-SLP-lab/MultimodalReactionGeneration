from argparse import Namespace
from typing import Any, Callable, Dict


def ext_args(target: Callable, args: Namespace) -> Dict[str, Any]:
    """Extract args from Namespace object.

    Args:
        target (object): target class object or instance, or function
        args (Namespace): Namespace object
    """

    if hasattr(target, "__init__"):
        target = target.__init__
    if not hasattr(target, "__code__"):
        raise ValueError("target must be class or function")

    args_list = target.__code__.co_varnames[: target.__code__.co_argcount]
    namespase_args = vars(args)
    arg_dict = {
        arg: namespase_args[arg]
        for arg in args_list
        if arg in namespase_args and arg != "self"
    }

    return arg_dict
