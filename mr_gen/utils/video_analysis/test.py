import cv2
from matplotlib.patheffects import Normal
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

from mr_gen.utils.tools.rotations import matrix_to_angles, angles_to_matrix, calc_R

from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


NormalizedLandmarkList.mro()


def head_pose_plotter(
    frame: np.ndarray, face: np.ndarray, angle: np.ndarray
) -> np.ndarray:
    R = angles_to_matrix(angle)

    head_direction = np.array([0, 0, -1]) * 200
    head_direction = np.dot(R, head_direction)

    for x, y, _ in face:
        res = _normalized_to_pixel_coordinates(x, y, frame.shape[1], frame.shape[0])
        if res is not None:
            x, y = res
        cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    return frame


face_mesh = FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True,
)

img = cv2.imread("mr_gen/utils/video_analysis/myface.jpg")

# Flip the image horizontally for a later selfie-view display
# Also convert the color space from BGR to RGB
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# To improve performance
image.flags.writeable = False

# Get the result
recognission = face_mesh.process(image.copy())
landmark = recognission.multi_face_landmarks[0].landmark  # type: ignore
print(landmark[0].x, landmark[0].y, landmark[0].z)
print(type(landmark[0]))
print(type(landmark))
lm: NormalizedLandmarkList = landmark[0]
print(lm.__dir__())
print(np.dot(np.array(landmark), np.array([0, 1, 0])))
# for lm in landmark:
#     print(lm.x, lm.y, lm.z)
# print(list(landmark[0]))

# To improve performance
image.flags.writeable = True

# Convert the color space from RGB to BGR
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

cv2.imwrite("myface2.png", img)
