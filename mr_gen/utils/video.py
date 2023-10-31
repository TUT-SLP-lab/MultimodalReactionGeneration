"""
(Writer):   _VideoB -> VideoIOBase -> _VideoWB -> VideoWriter
(Reader):   _VideoB -> VideoIOBase -> _VideoRB -> _VideoRIdx -> VideoReader
"""

from __future__ import annotations
import os
from abc import ABC, abstractmethod
import warnings
from typing import List, Optional, Union, Tuple, overload
from typing_extensions import Literal, TypeAlias
import cv2
from numpy import ndarray


VideoIOMode: TypeAlias = Literal["w", "r"]

ATTR_ALL = [
    "_set_writer",
    "_write_frame",
    "write",
    "read",
    "trime_time",
    "close",
    "__getitem__",
    "__len__",
    "__iter__",
    "__next__",
    "_reset",
    "_set_iter",
]


class _VideoB(ABC):
    def __init__(self) -> None:
        pass

    def _set_writer(self, _size_wh: Tuple[int, int]) -> None:
        raise AttributeError("This instance has no attribute _set_writer().")

    def _write_frame(self, _frame: ndarray) -> None:
        raise AttributeError("This instance has no attribute _write_frame().")

    def write(self, _frames: Union[ndarray, List[ndarray], _VideoRIdx]) -> None:
        raise AttributeError("This instance has no attribute write().")

    def read(self) -> ndarray:
        raise AttributeError("This instance has no attribute read().")

    def trime_time(self, _start: float, _stop: float) -> _VideoRIdx:
        raise AttributeError("This instance has no attribute trime_time().")

    @abstractmethod
    def close(self) -> None:
        pass

    def __getitem__(self, idx) -> Union[ndarray, _VideoRIdx]:
        raise AttributeError("This instance has no attribute __getitem__().")

    def __len__(self) -> int:
        raise AttributeError("This instance has no attribute __len__().")

    def __iter__(self) -> _VideoRIdx:
        raise AttributeError("This instance has no attribute __iter__().")

    def __next__(self) -> ndarray:
        raise AttributeError("This instance has no attribute __next__().")

    def _reset(self) -> None:
        raise AttributeError("This instance has no attribute _reset().")

    def _set_iter(self, _sss: slice) -> _VideoRIdx:
        raise AttributeError("This instance has no attribute _set_iter().")


class VideoIOBase(_VideoB):
    def __init__(self, path: str, codec: str = "mp4v") -> None:
        super().__init__()

        self._fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore
        self._codec = codec

        self._path = path
        self._name = os.path.basename(path)

        self._cap_width: int = 0
        self._cap_height: int = 0
        self._cap_frames: int = 0
        self._fps: float = 0.0

    def get_cap_width(self):
        return self._cap_width

    def get_cap_height(self):
        return self._cap_height

    def get_cap_frames(self):
        return self._cap_frames

    def get_fps(self):
        return self._fps

    def get_path(self):
        return self._path

    def get_codec(self):
        return self._codec

    def __del__(self):
        self.close()

    def __enter__(self) -> VideoIOBase:
        return self

    def __exit__(self, *_):
        self.close()


class _VideoWB(VideoIOBase):
    def __init__(self, path: str, fps: float, codec: str = "mp4v", **_) -> None:
        super().__init__(path, codec)

        self._writer = None
        self._fps = fps

    def _set_writer(self, size_wh: Tuple[int, int]) -> None:
        if self._writer is not None:
            self.close()
        self._cap_width, self._cap_height = size_wh
        self._writer = cv2.VideoWriter(self._path, self._fourcc, self._fps, size_wh)

    def _write_frame(self, frame: ndarray) -> None:
        if self._writer is None:
            raise ValueError("Writer is not set.")
        self._writer.write(frame)

    def close(self):
        if self._writer is not None:
            self._writer.release()
        del self


class VideoWriter(_VideoWB):
    def write(self, frames: Union[ndarray, List[ndarray], _VideoRIdx]):
        if isinstance(frames, ndarray):
            frames = [frames]
        if self._writer is None:
            size_wh = (frames[0].shape[1], frames[0].shape[0])
            self._set_writer(size_wh)
        for frame in frames:
            self._write_frame(frame)


class _VideoRB(VideoIOBase):
    def __init__(self, path: str, codec: str = "mp4v") -> None:
        super().__init__(path, codec)

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            warnings.warn("Video File is not opened.")

        self._cap_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._cap_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._cap_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)

    def close(self):
        self._cap.release()
        del self


class _VideoRIdx(_VideoRB):
    def __init__(self, path: str, codec: str = "mp4v", **kwargs) -> None:
        super().__init__(path, codec)

        self._current_idx = 0

        self._step = kwargs.get("step", 1)
        self._start = kwargs.get("start", 0)
        self._stop = kwargs.get("stop", self._cap_frames)

        # check args
        if not (isinstance(self._step, int) or self._step is None):
            raise ValueError("Invalid type of args: step")
        if not (isinstance(self._start, int) or self._start is None):
            raise ValueError("Invalid type of args: start")
        if not (isinstance(self._stop, int) or self._stop is None):
            raise ValueError("Invalid type of args: stop")
        # overwrite args
        if self._step is None:
            self._step = 1
        if self._start is None:
            self._start = 0
        if self._stop is None:
            self._stop = self._cap_frames

        self._length = self.__len__()

    @overload
    def __getitem__(self, idx: int) -> ndarray:
        ...

    @overload
    def __getitem__(self, idx: slice) -> _VideoRIdx:
        ...

    def __getitem__(self, idx) -> Union[ndarray, _VideoRIdx]:
        if isinstance(idx, int):
            pos = cv2.CAP_PROP_POS_FRAMES
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _, frame = self._cap.read()
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

            return frame
        elif isinstance(idx, slice):
            return self._set_iter(idx)
        else:
            raise IndexError("Invalid index.")

    def __len__(self) -> int:
        length = self._stop - self._start
        return length

    def __iter__(self) -> _VideoRIdx:
        return self

    def __next__(self) -> ndarray:
        if self._current_idx == self._length + self._start:
            self._reset()
            raise StopIteration

        frame = self._cap.read()[1]
        self._current_idx += self._step
        if self._step > 1:
            current_frame = self._start + self._current_idx
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        return frame

    def _reset(self):
        self._current_idx = self._start
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._start)

    def _set_iter(self, sss: slice) -> _VideoRIdx:
        return _VideoRIdx(
            path=self._path,
            codec=self._codec,
            step=sss.step,
            start=sss.start,
            stop=sss.stop,
        )


class VideoReader(_VideoRIdx):
    def read(self) -> ndarray:
        return self._cap.read()[1]

    def trime_time(self, start: float, stop: float) -> _VideoRIdx:
        start_frame = int(self._fps * start)
        stop_frame = int(self._fps * stop)

        return self.trime_frame(start_frame, stop_frame)

    def trime_frame(self, start: int, stop: int) -> _VideoRIdx:
        return self[start:stop:1]


def open_video(
    path: str, mode: VideoIOMode = "r", codec="mp4v", fps: Optional[float] = None
) -> VideoIOBase:
    if mode == "r":
        return VideoReader(path, codec)
    elif mode == "w":
        if fps is None:
            raise ValueError("when 'mode' is 'r', 'fps' must be specified.")
        return VideoWriter(path, fps, codec)
    else:
        raise ValueError("'mode' must be 'r' or 'w'.")
