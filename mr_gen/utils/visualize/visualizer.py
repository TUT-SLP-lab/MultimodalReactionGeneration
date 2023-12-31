from typing import Union, Dict, Tuple
import cv2
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
from numpy import ndarray
import numpy as np

from mr_gen.utils.tools import FaceAdapter
from mr_gen.utils.tools.rotations import angles_to_matrix


CUBE_SIZE = 600
CUBE = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
)
CUBE_CONTOURS = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
)
BOTTOM_CONTOURS = np.array([4, 5, 7, 6])

FACE_OVAL = np.array(
    [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ]
)


def head_pose_plotter(
    frame: ndarray,
    head_pose: Union[FaceAdapter, Dict[str, ndarray], None],
    clr: Tuple[int, int, int] = (0, 255, 0),
    clr_sub: Tuple[int, int, int] = (0, 0, 255),
):
    if head_pose is None:
        return frame
    shape = frame.shape

    if isinstance(head_pose, dict):
        angle = head_pose["angle"]
        centroid = head_pose["centroid"]
        face = head_pose["face"]
    if isinstance(head_pose, FaceAdapter):
        angle = head_pose.angle
        centroid = head_pose.centroid
        face = head_pose.face

    R = angles_to_matrix(angle)[0]

    head_direction = np.array([0, 0, 1]) * 200
    head_direction = np.dot(R, head_direction)[:2]

    face = np.dot(R.T, face.T).T + centroid
    nose_2d = face[1][:2]

    xy = _normalized_to_pixel_coordinates(nose_2d[0], nose_2d[1], shape[1], shape[0])
    if xy is not None:
        start_p = np.array((xy[0], xy[1]))
        stop_p = start_p + head_direction.astype(np.int32)
        cv2.line(frame, start_p, stop_p, clr_sub, 3)

    for x, y, _ in face:
        res = _normalized_to_pixel_coordinates(x, y, shape[1], shape[0])
        if res is not None:
            x, y = res
            cv2.circle(frame, (x, y), 1, clr, 1)

    return frame
