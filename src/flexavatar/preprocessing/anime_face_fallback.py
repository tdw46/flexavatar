from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np


ANIME_CASCADE_URL = "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml"
ANIME_CASCADE_PATH = Path.home() / ".cache" / "flexavatar" / "lbpcascade_animeface.xml"
ANIME_FACE_SCORE = 0.995

_fallback_used = False


def reset_anime_fallback_usage():
    global _fallback_used
    _fallback_used = False


def anime_fallback_was_used():
    return _fallback_used


def _ensure_anime_cascade():
    ANIME_CASCADE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ANIME_CASCADE_PATH.exists() or ANIME_CASCADE_PATH.stat().st_size == 0:
        urlretrieve(ANIME_CASCADE_URL, ANIME_CASCADE_PATH)
    return ANIME_CASCADE_PATH


def detect_anime_faces(image: np.ndarray):
    cascade_path = _ensure_anime_cascade()
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load anime face cascade: {cascade_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    detections = []
    for scale_factor, min_neighbors in ((1.1, 5), (1.05, 3), (1.03, 2)):
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(48, 48),
        )
        if len(faces) > 0:
            detections = faces
            break

    if len(detections) == 0:
        return []

    image_h, image_w = image.shape[:2]
    fallback_faces = []
    for x, y, w, h in detections:
        x = int(max(0, min(x, image_w - 1)))
        y = int(max(0, min(y, image_h - 1)))
        w = int(max(1, min(w, image_w - x)))
        h = int(max(1, min(h, image_h - y)))
        fallback_faces.append(["face", ANIME_FACE_SCORE, x, y, w, h])

    fallback_faces.sort(key=lambda det: det[4] * det[5], reverse=True)
    return fallback_faces


def _write_anime_segmentation_fallback(preprocessing_folder: str, video_name: str):
    out_dir = Path(preprocessing_folder) / video_name
    cropped_dir = out_dir / "cropped"
    seg_dir = out_dir / "seg_og"
    annot_dir = out_dir / "seg_non_crop_annotations"
    if not cropped_dir.exists():
        return

    seg_dir.mkdir(parents=True, exist_ok=True)
    annot_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = sorted(
        path for path in cropped_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for frame_path in frame_paths:
        seg_path = seg_dir / f"{frame_path.stem}.png"
        if seg_path.exists():
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        image_h, image_w = image.shape[:2]
        anime_faces = detect_anime_faces(image)
        if anime_faces:
            _, _, x, y, w, h = anime_faces[0]
        else:
            inset_x = int(image_w * 0.12)
            inset_y = int(image_h * 0.08)
            x, y = inset_x, inset_y
            w, h = image_w - 2 * inset_x, image_h - 2 * inset_y

        mask = np.zeros((image_h, image_w), dtype=np.uint8)
        center = (int(x + w / 2), int(y + h / 2))
        axes = (max(1, int(w * 0.47)), max(1, int(h * 0.55)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 2, thickness=-1)
        cv2.imwrite(str(seg_path), mask)

        annotation = np.zeros((image_h, image_w, 3), dtype=np.uint8)
        annotation[mask == 2] = (218, 173, 128)
        cv2.imwrite(str(annot_dir / f"color_{frame_path.stem}.png"), annotation)
        print(f"Pixel3DMM anime segmentation fallback wrote {seg_path}")


def _install_anime_segmentation_fallback(main_pixel3dmm):
    globals_dict = getattr(main_pixel3dmm, "__globals__", None)
    if not globals_dict or "main_facer" not in globals_dict:
        return
    current_main_facer = globals_dict["main_facer"]
    if getattr(current_main_facer, "_flexavatar_anime_segmentation_fallback", False):
        return

    original_main_facer = current_main_facer

    def anime_segmentation_fallback_main_facer(preprocessing_folder: str, video_name: str, /):
        original_main_facer(preprocessing_folder, video_name)
        if anime_fallback_was_used():
            _write_anime_segmentation_fallback(preprocessing_folder, video_name)

    anime_segmentation_fallback_main_facer._flexavatar_anime_segmentation_fallback = True
    globals_dict["main_facer"] = anime_segmentation_fallback_main_facer


def install_anime_face_fallback(main_pixel3dmm=None):
    import pixel3dmm.preprocessing.pipnet_utils as pipnet_utils

    current_detector = pipnet_utils.FaceBoxesDetector
    if getattr(current_detector, "_flexavatar_anime_fallback", False):
        if main_pixel3dmm is not None:
            _install_anime_segmentation_fallback(main_pixel3dmm)
        return

    original_detector = current_detector

    class AnimeFallbackFaceBoxesDetector:
        _flexavatar_anime_fallback = True

        def __init__(self, *args, **kwargs):
            self._original = original_detector(*args, **kwargs)

        def detect(self, image, thresh=0.6, im_scale=None):
            detections, scale = self._original.detect(image, thresh, im_scale)
            best_score = max(
                (float(det[1]) for det in detections if len(det) > 1 and det[0] == "face"),
                default=0.0,
            )
            if best_score >= 0.75:
                return detections, scale

            anime_detections = detect_anime_faces(image)
            if anime_detections:
                global _fallback_used
                _fallback_used = True
                best = anime_detections[0]
                print(
                    "Pixel3DMM face detector fallback: anime face "
                    f"bbox x={best[2]} y={best[3]} w={best[4]} h={best[5]}"
                )
                return anime_detections, 1

            return detections, scale

    pipnet_utils.FaceBoxesDetector = AnimeFallbackFaceBoxesDetector
    if main_pixel3dmm is not None:
        _install_anime_segmentation_fallback(main_pixel3dmm)
