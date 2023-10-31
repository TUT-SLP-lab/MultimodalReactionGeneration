import numpy as np
from typing import Iterable, List, NamedTuple, Union
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


def calc_R(lm: Union[NamedTuple, np.ndarray], img_w, img_h) -> np.ndarray:
    """
    Calculate rotation matrix.
    Rotate the FaceMesh by R to face the front
    """
    scale_vec = np.array([img_w, img_h, img_w])

    def extractor_namedtuple(lm: Iterable[NormalizedLandmarkList]) -> List[np.ndarray]:
        p33 = np.array([lm[33].x, lm[33].y, lm[33].z]) * scale_vec  # type: ignore
        p263 = np.array([lm[263].x, lm[263].y, lm[263].z]) * scale_vec  # type: ignore
        p152 = np.array([lm[152].x, lm[152].y, lm[152].z]) * scale_vec  # type: ignore
        p10 = np.array([lm[10].x, lm[10].y, lm[10].z]) * scale_vec  # type: ignore

        return [p33, p263, p152, p10]

    def extractor_ndarray(lm: np.ndarray) -> List[np.ndarray]:
        p33 = lm[33] * scale_vec
        p263 = lm[263] * scale_vec
        p152 = lm[152] * scale_vec
        p10 = lm[10] * scale_vec

        return [p33, p263, p152, p10]

    if isinstance(next(iter(lm)), NormalizedLandmarkList):
        p33, p263, p152, p10 = extractor_namedtuple(lm)
    elif isinstance(lm, np.ndarray):
        p33, p263, p152, p10 = extractor_ndarray(lm)
    else:
        raise ValueError("Invalid type.")

    _x = p263 - p33
    x = _x / np.linalg.norm(_x)

    _y = p152 - p10
    xy = x * np.dot(x, _y)
    y = _y - xy
    y = y / np.linalg.norm(y)

    z = np.cross(x, y)
    z = z / np.linalg.norm(y)

    R = np.array([x, y, z])

    return R


def matrix_to_angles(matrixes: Union[Iterable[np.ndarray], np.ndarray]) -> np.ndarray:
    """This 'matrixes' is a rotation matrix
    that rotates the face from the front to the actual direction
    """
    if isinstance(matrixes, np.ndarray):
        if matrixes.ndim == 2 and matrixes.shape == (3, 3):
            matrixes = [matrixes]
        elif matrixes.ndim == 3 and matrixes[1:] == (3, 3):
            pass
        else:
            raise ValueError("'matrixes' must be (*, 3, 3).")

    angles = []
    # This R is a rotation matrix that rotates the face from the front to the actual direction
    for R in matrixes:
        x, y, z = _rotation_angles(R)
        angles.append([x, y, z])  # 360 degree base

    return np.array(angles)


def angles_to_matrix(angles: Union[Iterable[np.ndarray], np.ndarray]) -> np.ndarray:
    """This 'angles' is a euler angle
    that rotates the face from the front to the actual direction
    """
    if isinstance(angles, np.ndarray):
        if angles.ndim == 1 and angles.shape == (3,):
            angles = [angles]
        elif angles.ndim == 2 and angles[0].shape == (3,):
            pass
        else:
            raise ValueError(
                f"'angles' must be (*, 3). ndim={angles.ndim}, shape={angles.shape}"
            )

    matrixes = []
    # This angle is a rotation matrix that rotates the face from the front to the actual direction
    for angle in angles:
        R = _rotation_matrix(*angle)
        matrixes.append(R)

    return np.stack(matrixes)


def _rotation_matrix(
    theta1: float, theta2: float, theta3: float, order="xyz"
) -> np.ndarray:
    """
    入力
        theta1, theta2, theta3 = Angle of rotation theta 1, 2, 3 in order of rotation
        oreder = Order of rotation e.g. 'xzy' for X, Z, Y order
    出力
        3x3 Rotation Matrix
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == "xzx":
        matrix = np.array(
            [
                [c2, -c3 * s2, s2 * s3],
                [c1 * s2, c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3],
                [s1 * s2, c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "xyx":
        matrix = np.array(
            [
                [c2, s2 * s3, c3 * s2],
                [s1 * s2, c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1],
                [-c1 * s2, c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yxy":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, s1 * s2, c1 * s3 + c2 * c3 * s1],
                [s2 * s3, c2, -c3 * s2],
                [-c3 * s1 - c1 * c2 * s3, c1 * s2, c1 * c2 * c3 - s1 * s3],
            ]
        )
    elif order == "yzy":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c1 * s2, c3 * s1 + c1 * c2 * s3],
                [c3 * s2, c2, s2 * s3],
                [-c1 * s3 - c2 * c3 * s1, s1 * s2, c1 * c3 - c2 * s1 * s3],
            ]
        )
    elif order == "zyz":
        matrix = np.array(
            [
                [c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                [-c3 * s2, s2 * s3, c2],
            ]
        )
    elif order == "zxz":
        matrix = np.array(
            [
                [c1 * c3 - c2 * s1 * s3, -c1 * s3 - c2 * c3 * s1, s1 * s2],
                [c3 * s1 + c1 * c2 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                [s2 * s3, c3 * s2, c2],
            ]
        )
    elif order == "xyz":
        matrix = np.array(
            [
                [c2 * c3, -c2 * s3, s2],
                [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2],
            ]
        )
    elif order == "xzy":
        matrix = np.array(
            [
                [c2 * c3, -s2, c2 * s3],
                [s1 * s3 + c1 * c3 * s2, c1 * c2, c1 * s2 * s3 - c3 * s1],
                [c3 * s1 * s2 - c1 * s3, c2 * s1, c1 * c3 + s1 * s2 * s3],
            ]
        )
    elif order == "yxz":
        matrix = np.array(
            [
                [c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3, c2 * s1],
                [c2 * s3, c2 * c3, -s2],
                [c1 * s2 * s3 - c3 * s1, c1 * c3 * s2 + s1 * s3, c1 * c2],
            ]
        )
    elif order == "yzx":
        matrix = np.array(
            [
                [c1 * c2, s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3],
                [s2, c2 * c3, -c2 * s3],
                [-c2 * s1, c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3],
            ]
        )
    elif order == "zyx":
        matrix = np.array(
            [
                [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
                [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
                [-s2, c2 * s3, c2 * c3],
            ]
        )
    elif order == "zxy":
        matrix = np.array(
            [
                [c1 * c3 - s1 * s2 * s3, -c2 * s1, c1 * s3 + c3 * s1 * s2],
                [c3 * s1 + c1 * s2 * s3, c1 * c2, s1 * s3 - c1 * c3 * s2],
                [-c2 * s3, s2, c2 * c3],
            ]
        )
    else:
        raise ValueError("Invalid order.")

    return matrix


def _rotation_angles(matrix: np.ndarray, order: str = "xyz") -> np.ndarray:
    """
    Parameters
        matrix = 3x3 Rotation Matrix
        oreder = Order of rotation e.g. 'xzy' for X, Z, Y order
    Outputs
        theta1, theta2, theta3 = Angle of rotation theta 1, 2, 3 in order of rotation
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    if order == "xzx":
        theta1 = np.arctan(r31 / r21)
        theta2 = np.arctan(r21 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(-r13 / r12)

    elif order == "xyx":
        theta1 = np.arctan(-r21 / r31)
        theta2 = np.arctan(-r31 / (r11 * np.cos(theta1)))
        theta3 = np.arctan(r12 / r13)

    elif order == "yxy":
        theta1 = np.arctan(r12 / r32)
        theta2 = np.arctan(r32 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(-r21 / r23)

    elif order == "yzy":
        theta1 = np.arctan(-r32 / r12)
        theta2 = np.arctan(-r12 / (r22 * np.cos(theta1)))
        theta3 = np.arctan(r23 / r21)

    elif order == "zyz":
        theta1 = np.arctan(r23 / r13)
        theta2 = np.arctan(r13 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(-r32 / r31)

    elif order == "zxz":
        theta1 = np.arctan(-r13 / r23)
        theta2 = np.arctan(-r23 / (r33 * np.cos(theta1)))
        theta3 = np.arctan(r31 / r32)

    elif order == "xzy":
        theta1 = np.arctan(r32 / r22)
        theta2 = np.arctan(-r12 * np.cos(theta1) / r22)
        theta3 = np.arctan(r13 / r11)

    elif order == "xyz":
        theta1 = np.arctan(-r23 / r33)
        theta2 = np.arctan(r13 * np.cos(theta1) / r33)
        theta3 = np.arctan(-r12 / r11)

    elif order == "yxz":
        theta1 = np.arctan(r13 / r33)
        theta2 = np.arctan(-r23 * np.cos(theta1) / r33)
        theta3 = np.arctan(r21 / r22)

    elif order == "yzx":
        theta1 = np.arctan(-r31 / r11)
        theta2 = np.arctan(r21 * np.cos(theta1) / r11)
        theta3 = np.arctan(-r23 / r22)

    elif order == "zyx":
        theta1 = np.arctan(r21 / r11)
        theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
        theta3 = np.arctan(r32 / r33)

    elif order == "zxy":
        theta1 = np.arctan(-r12 / r22)
        theta2 = np.arctan(r32 * np.cos(theta1) / r22)
        theta3 = np.arctan(-r31 / r33)
    else:
        raise ValueError("Invalid order.")

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi

    return np.array((theta1, theta2, theta3))
