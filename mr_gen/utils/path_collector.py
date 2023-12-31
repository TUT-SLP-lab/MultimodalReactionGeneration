from typing import Union, List, Dict, Callable, Optional
import dfcon
from cmpfilter import Filter, EmpFilter
from dfcon.path_filter import FileFilter


def mp4_collector(
    target_path: str,
    name: Union[None, str, List[str], Filter],
) -> List[str]:
    """MP4 path collector in dataset.

    Args:
        target_path (str): dataset root path.
        name: target mp4 file name. \n
            None: get all mp4 file. \n
            str: get mono mp4 file. \n
            List[str]: get specify mp4 file. \n
            dfcon.Filter: get only matching mp4 file.
    Returns:
        List[str]: serialized mp4 file path list.
    """
    ffilter = gen_file_filter("mp4", name)
    path_list = data_collector(target_path, ffilter)
    return path_list


def wav_collector(
    target_path: str,
    name: Union[None, str, List[str], Filter],
) -> List[str]:
    """WAV path collector in dataset.

    Args:
        target_path (str): dataset root path.
        name: target wav file name. \n
            None: get all wav file. \n
            str: get mono wav file. \n
            List[str]: get specify wav file. \n
            dfcon.Filter: get only matching wav file.
    Returns:
        List[str]: serialized wav file path list.
    """
    ffilter = gen_file_filter("wav", name)
    path_list = data_collector(target_path, ffilter)
    return path_list


def data_collector(target_path: str, ffilter: Optional[Filter] = None) -> List[str]:
    """File path collector in dataset.

    Args:
        target_path (str): dataset root path.
        name: target file name each extention. \n
            None: get all file. \n
            Dict[ext, ...]: \n
                None: get all file that specify extention. \n
                str: get mono file that specify extention. \n
                List[str]: get specify file that specify extention. \n
                dfcon.Filter: get only matching file that specify extention.
    """
    dirc = dfcon.Directory(path=target_path).build_structure()

    # make to Filter objects all `name` dictionary value
    if ffilter is None:
        ffilter = EmpFilter()

    return dirc.get_file_path(ffilter, serialize=True)


def pair_collector(
    target_path: str, grouping_key: Callable[[str], str]
) -> Dict[str, List[str]]:
    name_filter = FileFilter().include_extention(["wav", "mp4"])
    name_filter = name_filter.contained(["host", "comp"])

    dirc = dfcon.Directory(path=target_path).build_structure(name_filter)
    result: Dict[str, List[str]] = dirc.get_grouped_path_list(grouping_key)

    return result


def gen_file_filter(ext: str, name: Union[None, str, List[str], Filter]) -> Filter:
    ffilter = FileFilter().include_extention([ext])
    if name is not None and isinstance(name, str):
        ffilter = ffilter.contained(name)
    elif isinstance(name, Filter):
        ffilter = ffilter & name
    elif isinstance(name, list):
        ffilter = ffilter.contained(name)
    elif isinstance(name, type(None)):
        pass
    else:
        raise TypeError(f"Invalid type {type(name)}")
    return ffilter
