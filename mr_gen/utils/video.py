from typing import List
import math
import os
import cv2
from numpy import ndarray
from tqdm import tqdm
import warnings


class Video:
    def __init__(self, video_path: str, codec: str = "mp4v") -> None:
        self.cap = cv2.VideoCapture(video_path)
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        if not self.cap.isOpened():
            warnings.warn("Video File is not opened.")

        self.path = video_path
        self.name = os.path.basename(video_path)
        self.codec = codec

        self.cap_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.cap_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.cap_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.writer = None

        self.step = 1

        self.current_idx = 0

        self.length = None
        self.__len__()

    def __str__(self) -> str:
        return f"all frame : {self.cap_frames}, fps : {round(self.fps, 2)}, time : {round(self.cap_frames/self.fps, 2)}"

    def __getitem__(self, idx):
        pos = cv2.CAP_PROP_POS_FRAMES
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        return ret, frame

    def __len__(self) -> int:
        if self.length is None:
            self.length = math.ceil(self.cap_frames / self.step)
        return self.length

    def __iter__(self):
        return self

    def __next__(self) -> ndarray:
        if self.current_idx == self.length:
            self.reset()
            raise StopIteration

        frame = self.cap.read()[1]
        self.current_idx += 1
        for _ in range(self.step - 1):
            self.cap.read()
        return frame

    def reset(self):
        self.current_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def info(self):
        return [self.fourcc, self.cap_width, self.cap_height, self.fps, self.cap_frames]

    def read(self) -> ndarray:
        return self.cap.read()

    def set_out_path(self, path: str):
        self.writer = cv2.VideoWriter(
            path, self.fourcc, self.fps, (self.cap_width, self.cap_height)
        )

    def write(self, frames: List[ndarray]):
        if isinstance(frames, ndarray):
            frames = [frames]

        for frame in frames:
            self.writer.write(frame)

    def set_step(self, step):
        self.step = step

    def close_writer(self):
        self.writer.release()

    def trime_time(
        self, start: float, stop: float, out: str, use_tqdm=False
    ) -> List[ndarray]:
        start_frame = int(self.fps * start)
        stop_frame = int(self.fps * stop)

        return self.trime_frame(start_frame, stop_frame, out, use_tqdm)

    def trime_frame(
        self, start: int, stop: int, out: str, use_tqdm=False
    ) -> List[ndarray]:
        # resp = []
        self.set_out_path(out)

        pos = cv2.CAP_PROP_POS_FRAMES
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        it = range(start, min(stop + 1, self.cap_frames))

        if use_tqdm:
            iterator = tqdm(it, desc="write-MP4")
        else:
            iterator = it

        for _ in iterator:
            _, frame = self.cap.read()
            self.write(frame)
            # resp.append(frame)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self.close_writer()

        # return resp
