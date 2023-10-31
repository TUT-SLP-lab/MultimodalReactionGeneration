"""This code is for Head-Motion-Estimation."""

import os
from typing import List, Optional, Tuple
from typing_extensions import Literal, TypeAlias

import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh
from tqdm import tqdm

from mr_gen.utils.video import open_video
from mr_gen.utils import head_pose_plotter
from mr_gen.utils import parallel_luncher
from mr_gen.utils.io import write_head_pose
from mr_gen.utils.tools import collect_landmark, FaceAdapter


VisualizeMode: TypeAlias = Literal["all", "sample", "none"]


class HeadPoseEstimation:
    def __init__(
        self,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        estimate_fps: float = 25.0,
        redo=False,
    ) -> None:
        """
        Args:
            min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. See details in
                https://solutions.mediapipe.dev/face_mesh#min_detection_confidence.
            min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
                face landmarks to be considered tracked successfully. See details in
                https://solutions.mediapipe.dev/face_mesh#min_tracking_confidence.
            max_num_faces: Maximum number of faces to detect. See details in
                https://solutions.mediapipe.dev/face_mesh#max_num_faces.
            estimate_fps: Estimate fps. Defaults to 25.0.
            redo (bool, optional):
                Redo process when exist result file. Defaults to False.
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_num_face = 1
        self.estimate_fps = estimate_fps
        self.redo = redo

        self.detector_args = {
            "static_image_mode": False,
            "max_num_faces": self.max_num_face,
            "refine_landmarks": True,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }

    def __call__(
        self,
        io_path_list: List[Tuple[str, str]],
        pnum: int = 1,
        visualize: VisualizeMode = "none",
    ) -> List[str]:
        """Apply face-mesh to video.
        Args:
            io_path_list (List[Tuple[str, str]]): Tuple that input and output, path list
            pnum (int, optional): process number. Defaults to 1.
            visualize (bool, optional): visualize flag. Defaults to False.
        """

        results = []
        all_args = []

        for i, (video_path, output_path) in enumerate(io_path_list):
            if visualize == "all":
                visualize_mode = True
            elif visualize == "sample":
                visualize_mode = i % pnum == 0
            elif visualize == "none":
                visualize_mode = False
            else:
                raise ValueError(
                    f"visualize mode must be 'all', 'sample' or 'none'. not {visualize}"
                )

            all_args.append([video_path, output_path, i % pnum == 0, visualize_mode])

        results = parallel_luncher(self.apply_facemesh, all_args, pnum, unpack=True)

        return results

    def apply_facemesh(
        self,
        input_v: str,
        output_path: str,
        use_tq: bool = False,
        visualize: bool = False,
    ) -> str:
        recognizer = FaceMesh(**self.detector_args)
        vr = open_video(input_v, mode="r")

        skip_frame = vr.get_fps() / self.estimate_fps
        if skip_frame < 1:
            raise ValueError("fps of input video must be larger than estimate_fps")
        if skip_frame % 1 != 0:
            raise ValueError("fps of input video must be multiple of estimate_fps")
        skip_frame = int(skip_frame)

        results = []

        # When exist results & self.redo == False
        if os.path.isfile(output_path) and not self.redo:
            return output_path

        vw = None
        if visualize:
            visualize_path = output_path.rsplit(".", maxsplit=1)[0] + "_visualized.mp4"
            vw = open_video(visualize_path, mode="w", fps=self.estimate_fps)

        iterator = tqdm(vr, leave=False, desc="Est Lmark") if use_tq else vr
        for i, frame in enumerate(iterator):
            if i % skip_frame != 0:
                continue

            face_info = self.estimation(frame, recognizer)
            results.append(face_info)

            if vw is not None:
                frame = head_pose_plotter(frame, face_info)
                vw.write(frame)

        if vw is not None:
            vw.close()
        vr.close()

        write_head_pose(output_path, results)

        return output_path

    def estimation(
        self, frame: np.ndarray, face_mesh: FaceMesh
    ) -> Optional[FaceAdapter]:
        img_h, img_w, _ = frame.shape

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        recognission = face_mesh.process(image.copy())

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        result = collect_landmark(recognission, img_h, img_w)[0]
        if result is None:
            return None

        return result
