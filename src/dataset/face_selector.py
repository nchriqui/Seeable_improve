from typing import Dict, Optional
import numpy as np
from src.utils.bbox import bboxes_intersection_over_union, landmarks_to_bbox
from src.utils.types import convert_to_numpy_2D
from src.utils.types import BboxArray, LandmarksArray


def compute_intersection_stats(
    frame: Dict,
    detector_face: str = "retina",
    detector_landmarks: str = "dlib68",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Dict]:
    """matrix iou: row: detector_face col:detector_landmarks"""
    dect_lands = frame.get(detector_landmarks, [])
    dect_faces = frame.get(detector_face, [])
    if len(dect_lands) == 0 or len(dect_faces) == 0:
        return None

    # convert to numpy
    bboxes_face = convert_to_numpy_2D([d["bbox"] for d in dect_faces])
    bboxes_land = convert_to_numpy_2D(
        [landmarks_to_bbox(np.array(d["points"])) for d in dect_lands]
    )

    if False:
        # clip negative values
        bboxes_face = np.maximum(bboxes_face, 0)
        # bboxes_land = np.maximum(bboxes_land, 0)

        # clip bbox to image size
        print(width, height)
        if width is not None:
            bboxes_face[:, 0::2] = np.minimum(bboxes_face[:, 0::2], width - 1)
            # bboxes_land[:, 1::2] = np.minimum(bboxes_land[:, 1::2], width - 1)
        if height is not None:
            bboxes_face[:, 1::2] = np.minimum(bboxes_face[:, 1::2], height - 1)
            # bboxes_land[:, 1::2] = np.minimum(bboxes_land[:, 1::2], height - 1)

    if 1:
        W = width - 1 if width is not None else None
        H = height - 1 if height is not None else None
        bboxes_face = np.clip(np.rint(bboxes_face), 0, [W, H, W, H]).astype(int)
        bboxes_land = np.clip(np.rint(bboxes_land), 0, [W, H, W, H]).astype(int)

    iou_dict = bboxes_intersection_over_union(bboxes_face, bboxes_land)

    return {
        "bbox_face": bboxes_face,
        "bbox_land": bboxes_land,
        "N_land": bboxes_face.shape[0],
        "N_face": bboxes_land.shape[0],
        "iou_mat_face_land": iou_dict["iou_matrix"],
        "inter_mat_face_land": iou_dict["inter_matrix"],
        "union_mat_face_land": iou_dict["union_matrix"],
        "area_face": iou_dict["area1"],
        "area_land": iou_dict["area2"],
    }


def select_principal_face(
    frame: Dict,
    detector_face: str = "retina",
    detector_landmarks: str = "dlib68",
    largest_landmarks: bool = True,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Optional[Dict]:
    """Return landmarks and bbox from frame by computing the max iou between
    - the largest retina detected face bbox
    - all the landmarks' bbox
    """

    stats = compute_intersection_stats(
        frame,
        detector_face=detector_face,
        detector_landmarks=detector_landmarks,
        width=width,
        height=height,
    )
    if stats is None:
        return None

    # compute largest bbox for Face and Landmarks
    i_face_max = np.argmax(stats["area_face"])
    i_land_max = np.argmax(stats["area_land"])

    # face detector is the principal face
    # or landmark detector is the principal face
    if largest_landmarks:
        i_land = i_land_max
        i_face = np.argmax(stats["iou_mat_face_land"][:, i_land])
    else:
        i_face = i_face_max
        i_land = np.argmax(stats["iou_mat_face_land"][i_face])

    iou = stats["iou_mat_face_land"][i_face, i_land]
    if np.isclose(iou, 0):
        return None

    bbox = stats["bbox_face"][i_face]
    bbox_landmarks = stats["bbox_land"][i_land]
    landmarks = np.array(frame[detector_landmarks][i_land]["points"])
    keypoints = np.array(frame[detector_face][i_face]["points"])

    return {
        "bbox": bbox.view(BboxArray),
        "bbox_landmarks": bbox_landmarks.view(BboxArray),
        "landmarks": landmarks.view(LandmarksArray),
        "keypoints": keypoints.view(LandmarksArray),
        "iou": iou,
    }
