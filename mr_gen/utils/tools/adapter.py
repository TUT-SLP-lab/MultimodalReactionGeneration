from typing import Iterable, List, NamedTuple, Optional, Tuple
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList

from mr_gen.utils.tools.rotations import calc_R, matrix_to_angles


class FaceAdapter:
    def __init__(
        self, face: Iterable[NormalizedLandmarkList], img_h: int, img_w: int
    ) -> None:
        if isinstance(next(iter(face)), NormalizedLandmarkList):
            raise ValueError("Invalid type of args: face")
        if not hasattr(next(iter(face)), "x"):
            raise ValueError("Invalid type of args: face")
        if not hasattr(next(iter(face)), "y"):
            raise ValueError("Invalid type of args: face")
        if not hasattr(next(iter(face)), "z"):
            raise ValueError("Invalid type of args: face")

        self.resolution = (img_w, img_h)
        self.face = self.face2ndarray(face)
        self.nose = self.face[1].copy()
        self.centroid = self.face.mean(axis=0)
        self.face -= self.centroid
        self.angle, self.R = self.set_angle(self.face)

        self.face = np.dot(self.R, self.face.T).T

    def face2ndarray(self, face: Iterable[NormalizedLandmarkList]) -> np.ndarray:
        np_face = []

        for lm in face:
            np_face.append([lm.x, lm.y, lm.z])  # type: ignore

        return np.array(np_face)

    def set_angle(self, face: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        R = calc_R(face, *self.resolution)
        angle = matrix_to_angles(R)[0]

        return angle, R


def collect_landmark(
    recognission: NamedTuple, img_h: int, img_w: int
) -> List[Optional[FaceAdapter]]:
    if not hasattr(recognission, "multi_face_landmarks"):
        return [None]
    multi_face_landmarks = getattr(recognission, "multi_face_landmarks")
    if not multi_face_landmarks:
        return [None]

    landmarks = []
    for face in multi_face_landmarks:
        landmarks.append(FaceAdapter(getattr(face, "landmark"), img_h, img_w))

    return landmarks
