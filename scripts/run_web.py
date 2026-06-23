import asyncio
import hashlib
import io
import os
import ssl
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from shutil import copy2, rmtree
from threading import RLock, Thread
from typing import Optional

import cv2
import certifi
import numpy as np
import torch
from dreifus.vector.vector_torch import to_homogeneous
from elias.util import ensure_directory_exists_for_file, load_img, save_img
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from gaussian_splatting.gaussian_renderer import render_gsplat
from gaussian_splatting.scene import GaussianModel
from pixel3dmm.scripts.run_pixel3dmm import main as main_pixel3dmm
from pixel3dmm.utils.utils_3d import matrix_to_rotation_6d, rotation_6d_to_matrix
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from scipy.spatial.transform import Rotation as R

from flexavatar.config.dataset_config import SampleMetadata
from flexavatar.data_adapter.example_data import create_example_batch
from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
from flexavatar.data_adapter.nersemble_data_adapter import NeRSembleDataAdapter
from flexavatar.env import (
    FLEXAVATAR_AVATAR_CODE_PATH,
    FLEXAVATAR_INPUTS_PATH,
    FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH,
    REPO_ROOT,
)
from flexavatar.model.flexavatar_preprocessor import FlexAvatarPreprocessor
from flexavatar.model.sheap import SheapModule
from flexavatar.model_manager.avatar_code_manager import AvatarCodeManager
from flexavatar.model_manager.flexavatar_model_manager import FlexAvatarModelManager
from flexavatar.preprocessing.anime_face_fallback import (
    anime_fallback_was_used,
    detect_anime_faces,
    install_anime_face_fallback,
    reset_anime_fallback_usage,
)
from flexavatar.util.codes import interpolate_codes


def configure_certifi_ssl():
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    def create_certifi_context(*args, **kwargs):
        if "cafile" not in kwargs:
            kwargs["cafile"] = certifi.where()
        return ssl.create_default_context(*args, **kwargs)

    ssl._create_default_https_context = create_certifi_context


configure_certifi_ssl()


def run_pixel3dmm(image_path: str):
    image_name = Path(image_path).stem
    tracking_path = (
        f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/itw/{image_name}/"
        "tracking_nV1_noPho_uv2000.0_n1000.0/result.mp4"
    )
    if Path(tracking_path).exists():
        print(f"[Skipping] {image_name} because Pixel3DMM tracking already exists")
        return

    data_adapter = InTheWildDataAdapter(image_name)
    source_path = data_adapter.get_image_path(image_name)
    pixel3dmm_input_folder = f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/input/itw/{image_name}"
    is_video = source_path.endswith(".mp4")

    if is_video:
        pixel3dmm_input_path = f"{pixel3dmm_input_folder}/{image_name}.mp4"
        ensure_directory_exists_for_file(pixel3dmm_input_path)
        copy2(source_path, pixel3dmm_input_path)
    else:
        pixel3dmm_input_path = f"{pixel3dmm_input_folder}/{image_name}.jpg"
        ensure_directory_exists_for_file(pixel3dmm_input_path)
        image = load_img(source_path)
        save_img(image[..., :3], pixel3dmm_input_path)

    try:
        reset_anime_fallback_usage()
        install_anime_face_fallback(main_pixel3dmm)
        try:
            main_pixel3dmm(
                pixel3dmm_input_path,
                f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/processing/itw",
                f"{FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH}/tracking/itw",
                cleanup=True,
            )
            if anime_fallback_was_used() and os.environ.get("FLEXAVATAR_ALLOW_ANIME_FLAME_FALLBACK") != "1":
                raise RuntimeError(
                    "Anime face fallback detected a stylized face, but the current "
                    "Pixel3DMM/FlexAvatar path does not preserve anime style; it converts "
                    "the input toward a photoreal FLAME face. Set "
                    "FLEXAVATAR_ALLOW_ANIME_FLAME_FALLBACK=1 only for experimental tests."
                )
        except IndexError as exc:
            if anime_fallback_was_used():
                raise RuntimeError(
                    "Anime face fallback detected a face, but Pixel3DMM/FLAME tracking "
                    f"could not fit it into a usable avatar: {exc}"
                ) from exc
            raise RuntimeError(
                "Pixel3DMM did not detect a supported face in this image. "
                "Try a front-facing photoreal portrait, or load an existing avatar code."
            ) from exc
        except Exception as exc:
            if "does not preserve anime style" in str(exc):
                raise
            if anime_fallback_was_used():
                raise RuntimeError(
                    "Anime face fallback detected a face, but Pixel3DMM/FLAME tracking "
                    f"could not fit it into a usable avatar: {exc}"
                ) from exc
            raise
    finally:
        if Path(pixel3dmm_input_folder).is_dir():
            rmtree(pixel3dmm_input_folder)


def transform_gaussian_model(gaussian_model: GaussianModel, rigid_transform: torch.Tensor):
    transformed_xyz = (to_homogeneous(gaussian_model._xyz) @ rigid_transform.float().T)[..., :3]
    gaussian_model._xyz = transformed_xyz
    rotation_matrices = quaternion_to_matrix(gaussian_model._rotation)
    rotation_matrices_unposed = (rigid_transform[:3, :3] @ rotation_matrices.T).T
    gaussian_model._rotation = matrix_to_quaternion(rotation_matrices_unposed)


def projection_from_intrinsics(
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    near: float = 0.01,
    far: float = 10,
    z_sign: int = 1,
):
    h, w = image_size
    fx, fy, cx, cy = intrinsics
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = fx * 2 / w
    projection[1, 1] = fy * 2 / h
    projection[0, 2] = (w - 2 * cx) / w
    projection[1, 2] = (h - 2 * cy) / h
    projection[2, 2] = z_sign * (far + near) / (far - near)
    projection[2, 3] = -2 * far * near / (far - near)
    projection[3, 2] = z_sign
    return projection


@dataclass
class WebCamera:
    image_width: int = 1280
    image_height: int = 720
    radius: float = 1.0
    fovy: float = 20.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    @property
    def intrinsics(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.image_width / 2, self.image_height / 2], dtype=np.float32)

    @property
    def fovx(self):
        focal = self.image_height / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.degrees(2 * np.arctan(self.image_width / (2 * focal)))

    @property
    def pose(self):
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] += self.radius
        rotation = R.from_euler("yxz", [self.yaw, self.pitch, self.roll], degrees=True).as_matrix()
        rot_pose = np.eye(4, dtype=np.float32)
        rot_pose[:3, :3] = rotation
        pose = rot_pose @ pose
        pose[:, [1, 2]] *= -1
        return pose

    @property
    def world_view_transform(self):
        return np.linalg.inv(self.pose)

    @property
    def full_proj_transform(self):
        projection = projection_from_intrinsics(self.intrinsics, (self.image_height, self.image_width))
        return projection @ self.world_view_transform


class ControlPayload(BaseModel):
    mode: str = "default"
    playing: bool = True
    lock_head: bool = False
    expression: list[float] = []
    jaw: list[float] = [0, 0, 0]
    head: list[float] = [0, 0, 0]
    camera: dict = {}


@dataclass
class PanicAnimeAsset:
    name: str
    source_path: Path
    image_path: Path
    tha4_image_path: Path
    face_bbox: Optional[tuple[int, int, int, int]]
    image_offset: tuple[int, int]
    image_size: tuple[int, int]


class Tha4ExpressionHandler:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        src_path = repo_path / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        import tha4.poser.modes.mode_07 as mode_07
        from tha4.image_util import convert_output_image_from_torch_to_numpy, resize_PIL_image
        from tha4.poser.modes.pose_parameters import get_pose_parameters
        from tha4.shion.base.image_util import numpy_srgb_to_linear

        self._convert_output = convert_output_image_from_torch_to_numpy
        self._srgb_to_linear = numpy_srgb_to_linear
        self._resize_image = resize_PIL_image
        self._pose_parameters = get_pose_parameters()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        module_files = {
            "eyebrow_decomposer": str(repo_path / "data" / "tha4" / "eyebrow_decomposer.pt"),
            "eyebrow_morphing_combiner": str(repo_path / "data" / "tha4" / "eyebrow_morphing_combiner.pt"),
            "face_morpher": str(repo_path / "data" / "tha4" / "face_morpher.pt"),
            "body_morpher": str(repo_path / "data" / "tha4" / "body_morpher.pt"),
            "upscaler": str(repo_path / "data" / "tha4" / "upscaler.pt"),
        }
        self.poser = mode_07.create_poser(self.device, module_file_names=module_files, default_output_index=0)
        self.dtype = self.poser.get_dtype()
        self._render_lock = RLock()
        self.poser.get_modules()
        self._source_cache: dict[Path, torch.Tensor] = {}
        self._default_pose = self._build_default_pose()

    def render(
        self,
        source_path: Path,
        expression: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        wave: float,
    ):
        source = self._load_source(source_path)
        pose = self._default_pose.clone()
        self._set_symmetric_pose(pose, "eyebrow_raised", self._positive(expression, 0))
        self._set_symmetric_pose(pose, "eyebrow_lowered", max(self._negative(expression, 0), self._positive(expression, 21)))
        self._set_symmetric_pose(pose, "eyebrow_angry", self._positive(expression, 1))
        self._set_symmetric_pose(pose, "eyebrow_troubled", self._positive(expression, 2))
        self._set_symmetric_pose(pose, "eyebrow_happy", self._positive(expression, 3))
        self._set_symmetric_pose(pose, "eyebrow_serious", self._positive(expression, 20))
        self._set_symmetric_pose(pose, "mouth_raised_corner", self._positive(expression, 8))
        self._set_pose(pose, "mouth_lowered_corner_left", self._positive(expression, 9))
        self._set_pose(pose, "mouth_lowered_corner_right", self._positive(expression, 9))
        self._set_pose(pose, "mouth_aaa", self._positive(expression, 4))
        self._set_pose(pose, "mouth_iii", self._positive(expression, 5))
        self._set_pose(pose, "mouth_uuu", self._positive(expression, 6))
        self._set_pose(pose, "mouth_ooo", self._positive(expression, 7))
        self._set_pose(pose, "mouth_delta", self._positive(expression, 19))
        self._set_symmetric_pose(pose, "eye_relaxed", self._positive(expression, 10))
        self._set_symmetric_pose(pose, "eye_happy_wink", self._positive(expression, 11))
        self._set_symmetric_pose(pose, "eye_surprised", self._positive(expression, 12))
        self._set_symmetric_pose(pose, "eye_unimpressed", self._positive(expression, 13))
        self._set_symmetric_pose(pose, "eye_raised_lower_eyelid", self._positive(expression, 14))
        self._set_symmetric_pose(pose, "iris_small", self._positive(expression, 15))
        self._set_pose(pose, "mouth_smirk", self._positive(expression, 16))
        self._set_pose(pose, "iris_rotation_x", self._signed(expression, 17))
        self._set_pose(pose, "iris_rotation_y", self._signed(expression, 18))
        self._set_pose(pose, "head_x", self._clamp(-pitch / 24.0, -1.0, 1.0))
        self._set_pose(pose, "head_y", self._clamp(yaw / 32.0, -1.0, 1.0))
        self._set_pose(pose, "neck_z", self._clamp(roll / 18.0, -1.0, 1.0))
        self._set_pose(pose, "body_y", self._clamp(yaw / 48.0, -1.0, 1.0) * 0.35)
        self._set_pose(pose, "body_z", self._clamp(roll / 32.0, -1.0, 1.0) * 0.35)
        self._set_pose(pose, "breathing", self._clamp(0.45 + wave * 0.25, 0.0, 1.0))

        with self._render_lock, torch.no_grad():
            output_image = self.poser.pose(source, pose)[0].detach().cpu()
        array = self._convert_output(output_image)
        return Image.fromarray(array)

    def _load_source(self, source_path: Path):
        source_path = source_path.resolve()
        cached = self._source_cache.get(source_path)
        if cached is not None:
            return cached
        image = Image.open(source_path).convert("RGBA")
        image = self._resize_image(image, (self.poser.get_image_size(), self.poser.get_image_size()))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array[:, :, 0:3] = self._srgb_to_linear(array[:, :, 0:3])
        array = array * 2.0 - 1.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).float().to(self.device).to(self.dtype)
        self._source_cache[source_path] = tensor
        return tensor

    def _set_pose(self, pose: torch.Tensor, name: str, value: float):
        pose[self._pose_parameters.get_parameter_index(name)] = float(value)

    def _set_symmetric_pose(self, pose: torch.Tensor, group_name: str, value: float):
        self._set_pose(pose, f"{group_name}_left", value)
        self._set_pose(pose, f"{group_name}_right", value)

    def _build_default_pose(self):
        pose = torch.zeros(self.poser.get_num_parameters(), device=self.device, dtype=self.dtype)
        for group in self._pose_parameters.get_pose_parameter_groups():
            default_value = float(group.get_default_value())
            if default_value == 0.0:
                continue
            for offset in range(group.get_arity()):
                pose[group.get_parameter_index() + offset] = default_value
        return pose

    def _positive(self, expression: list[float], index: int):
        value = expression[index] if len(expression) > index else 0.0
        return self._clamp(value / 2.0, 0.0, 1.0)

    def _negative(self, expression: list[float], index: int):
        value = expression[index] if len(expression) > index else 0.0
        return self._clamp(-value / 2.0, 0.0, 1.0)

    def _signed(self, expression: list[float], index: int):
        value = expression[index] if len(expression) > index else 0.0
        return self._clamp(value, -1.0, 1.0)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float):
        return max(minimum, min(maximum, float(value)))


class AnimeSuperResolution:
    def __init__(self, model_path: Path, max_input_side: int = 256, cache_limit: int = 96):
        import onnxruntime as ort

        self.model_path = model_path
        self.max_input_side = max(64, int(max_input_side))
        self.cache_limit = max(8, int(cache_limit))
        self.blend_strength = self._clamp(float(os.environ.get("FLEXAVATAR_ANIME_SR_BLEND", "0.68")), 0.0, 1.0)
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anime-sr")
        self._lock = RLock()
        self._cache: dict[str, Image.Image] = {}
        self._pending: set[str] = set()
        self._cache_order: list[str] = []

    def upscale_cached(self, image: Image.Image, cache_key: Optional[str] = None):
        source = image.convert("RGBA")
        key = cache_key or self._cache_key(source)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached.copy(), True
            if key not in self._pending:
                self._pending.add(key)
                self._executor.submit(self._upscale_worker, key, source.copy())
        return source, False

    def cached(self, image: Image.Image, cache_key: Optional[str] = None):
        source = image.convert("RGBA")
        key = cache_key or self._cache_key(source)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached.copy(), True
        return source, False

    def cached_by_key(self, cache_key: str):
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        return None

    def warm(self, image: Image.Image, cache_key: Optional[str] = None):
        source = image.convert("RGBA")
        key = cache_key or self._cache_key(source)
        with self._lock:
            if key in self._cache or key in self._pending:
                return
            self._pending.add(key)
        self._upscale_worker(key, source.copy())

    def stats(self):
        with self._lock:
            return {
                "cached": len(self._cache),
                "pending": len(self._pending),
                "limit": self.cache_limit,
            }

    def _upscale_worker(self, key: str, image: Image.Image):
        try:
            result = self._upscale(image)
            with self._lock:
                self._cache[key] = result
                self._cache_order.append(key)
                while len(self._cache_order) > self.cache_limit:
                    old_key = self._cache_order.pop(0)
                    self._cache.pop(old_key, None)
        except Exception:
            traceback.print_exc()
        finally:
            with self._lock:
                self._pending.discard(key)

    def _upscale(self, image: Image.Image):
        resized = self._resize_for_model(image)
        padded = self._pad_for_model(resized)
        rgb = np.asarray(padded.convert("RGB"), dtype=np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)
        output = self._session.run([self._output_name], {self._input_name: tensor})[0][0]
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0.0, 1.0)
        rgb_image = Image.fromarray((output * 255.0 + 0.5).astype(np.uint8))
        scale_x = rgb_image.width / max(1, padded.width)
        scale_y = rgb_image.height / max(1, padded.height)
        target_size = (
            max(1, int(round(resized.width * scale_x))),
            max(1, int(round(resized.height * scale_y))),
        )
        rgb_image = rgb_image.crop((0, 0, target_size[0], target_size[1]))
        if self.blend_strength < 1.0:
            baseline = resized.convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
            rgb_image = Image.blend(baseline, rgb_image, self.blend_strength)
        alpha = resized.getchannel("A").resize(target_size, Image.Resampling.LANCZOS)
        return Image.merge("RGBA", (*rgb_image.split(), alpha))

    def _resize_for_model(self, image: Image.Image):
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_input_side:
            return image
        scale = self.max_input_side / longest
        next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(next_size, Image.Resampling.LANCZOS)

    def _pad_for_model(self, image: Image.Image):
        multiple = max(1, int(os.environ.get("FLEXAVATAR_ANIME_SR_MULTIPLE", "8")))
        width, height = image.size
        padded_width = int(np.ceil(width / multiple) * multiple)
        padded_height = int(np.ceil(height / multiple) * multiple)
        if (padded_width, padded_height) == image.size:
            return image
        padded = Image.new("RGBA", (padded_width, padded_height), (255, 255, 255, 0))
        padded.alpha_composite(image, (0, 0))
        return padded

    def _cache_key(self, image: Image.Image):
        digest = hashlib.sha1()
        digest.update(str(self.model_path).encode("utf-8"))
        digest.update(str(self.max_input_side).encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_SR_MULTIPLE", "8").encode("ascii"))
        digest.update(f"{self.blend_strength:.3f}".encode("ascii"))
        digest.update(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float):
        return max(minimum, min(maximum, float(value)))


class RealCuganSuperResolution:
    def __init__(
        self,
        executable_path: Path,
        model_dir: Path,
        max_input_side: int = 512,
        scale: int = 4,
        denoise: int = -1,
        tta: bool = False,
        tile_size: int = 0,
        syncgap: int = 3,
        jobs: str = "1:2:2",
        cache_limit: int = 96,
    ):
        self.executable_path = executable_path.resolve()
        self.model_dir = model_dir.resolve()
        self.max_input_side = max(64, int(max_input_side))
        self.scale = max(1, min(4, int(scale)))
        self.denoise = max(-1, min(3, int(denoise)))
        self.tta = bool(tta)
        self.tile_size = max(0, int(tile_size))
        self.syncgap = max(0, min(3, int(syncgap)))
        self.jobs = str(jobs)
        self.cache_limit = max(8, int(cache_limit))
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anime-realcugan")
        self._lock = RLock()
        self._cache: dict[str, Image.Image] = {}
        self._pending: set[str] = set()
        self._cache_order: list[str] = []

    def upscale_cached(self, image: Image.Image, cache_key: Optional[str] = None):
        source = image.convert("RGBA")
        key = cache_key or self._cache_key(source)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached.copy(), True
            if key not in self._pending:
                self._pending.add(key)
                self._executor.submit(self._upscale_worker, key, source.copy())
        return source, False

    def cached(self, image: Image.Image, cache_key: Optional[str] = None):
        source = image.convert("RGBA")
        key = cache_key or self._cache_key(source)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached.copy(), True
        return source, False

    def cached_by_key(self, cache_key: str):
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        return None

    def warm(self, image: Image.Image, cache_key: Optional[str] = None):
        source = image.convert("RGBA")
        key = cache_key or self._cache_key(source)
        with self._lock:
            if key in self._cache or key in self._pending:
                return
            self._pending.add(key)
        self._upscale_worker(key, source.copy())

    def stats(self):
        with self._lock:
            return {
                "cached": len(self._cache),
                "pending": len(self._pending),
                "limit": self.cache_limit,
            }

    def _upscale_worker(self, key: str, image: Image.Image):
        try:
            result = self._upscale(image)
            with self._lock:
                self._cache[key] = result
                self._cache_order.append(key)
                while len(self._cache_order) > self.cache_limit:
                    old_key = self._cache_order.pop(0)
                    self._cache.pop(old_key, None)
        except Exception:
            traceback.print_exc()
        finally:
            with self._lock:
                self._pending.discard(key)

    def _upscale(self, image: Image.Image):
        resized = self._resize_for_model(image)
        rgb_input = Image.new("RGBA", resized.size, (248, 252, 252, 255))
        rgb_input.alpha_composite(resized)

        with tempfile.TemporaryDirectory(prefix="flexavatar-realcugan-") as tmp:
            tmp_dir = Path(tmp)
            input_path = tmp_dir / "input.png"
            output_path = tmp_dir / "output.png"
            rgb_input.convert("RGB").save(input_path)
            model_dir_arg = str(self.model_dir)
            try:
                model_dir_arg = str(self.model_dir.relative_to(self.executable_path.parent))
            except ValueError:
                pass
            command = [
                str(self.executable_path),
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "-s",
                str(self.scale),
                "-n",
                str(self.denoise),
                "-t",
                str(self.tile_size),
                "-c",
                str(self.syncgap),
                "-j",
                self.jobs,
                "-m",
                model_dir_arg,
                "-f",
                "png",
            ]
            if self.tta:
                command.append("-x")
            completed = subprocess.run(
                command,
                cwd=str(self.executable_path.parent),
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "Real-CUGAN failed").strip()
                raise RuntimeError(detail)
            rgb_image = Image.open(output_path).convert("RGB")

        alpha = resized.getchannel("A").resize(rgb_image.size, Image.Resampling.LANCZOS)
        return Image.merge("RGBA", (*rgb_image.split(), alpha))

    def _resize_for_model(self, image: Image.Image):
        width, height = image.size
        longest = max(width, height)
        if longest <= self.max_input_side:
            return image
        scale = self.max_input_side / longest
        next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return image.resize(next_size, Image.Resampling.LANCZOS)

    def _cache_key(self, image: Image.Image):
        digest = hashlib.sha1()
        digest.update(str(self.executable_path).encode("utf-8"))
        digest.update(str(self.model_dir).encode("utf-8"))
        digest.update(str(self.max_input_side).encode("ascii"))
        digest.update(str(self.scale).encode("ascii"))
        digest.update(str(self.denoise).encode("ascii"))
        digest.update(str(int(self.tta)).encode("ascii"))
        digest.update(str(self.tile_size).encode("ascii"))
        digest.update(str(self.syncgap).encode("ascii"))
        digest.update(self.jobs.encode("ascii"))
        digest.update(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest()


class PanicAnimeBackend:
    """Adapter boundary for the anime reconstruction path.

    PAniC-3D is not a drop-in FLAME/FlexAvatar avatar. This backend keeps the
    routing, driver contract, and UI behavior separate while real PAniC-3D model
    installation/reconstruction can be added behind the same API.
    """

    def __init__(self):
        self.assets_dir = Path(REPO_ROOT) / "data" / "panic3d_assets"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.repo_path = Path(os.environ.get("PANIC3D_REPO_PATH", Path(REPO_ROOT) / "external" / "panic3d-anime-reconstruction"))
        self.tha4_repo_path = Path(os.environ.get("THA4_REPO_PATH", Path(REPO_ROOT) / "external" / "talking-head-anime-4-demo"))
        self.realcugan_dir = Path(os.environ.get(
            "FLEXAVATAR_REALCUGAN_DIR",
            Path(REPO_ROOT) / "external" / "realcugan-ncnn-vulkan-20220728-windows",
        ))
        self.realcugan_executable = Path(os.environ.get(
            "FLEXAVATAR_REALCUGAN_EXE",
            self.realcugan_dir / "realcugan-ncnn-vulkan.exe",
        ))
        self.realcugan_model_dir = Path(os.environ.get(
            "FLEXAVATAR_REALCUGAN_MODEL_DIR",
            self.realcugan_dir / os.environ.get("FLEXAVATAR_REALCUGAN_MODEL", "models-se"),
        ))
        self.adore_model_path = Path(os.environ.get(
            "FLEXAVATAR_ADORE_SR_MODEL",
            Path(REPO_ROOT) / "external" / "re-sisr" / "adore" / "2x_Adore_renarchi_fp32_onnxslim.onnx",
        ))
        self.sr_model_path = Path(os.environ.get(
            "FLEXAVATAR_ANIME_SR_MODEL",
            Path(REPO_ROOT) / "external" / "real-esrgan" / "models" / "RealESRGAN_x4plus_anime_6B.onnx",
        ))
        self._tha4_live_handler = None
        self._tha4_warmup_handler = None
        self._sr_handler = None
        self._sr_warmup_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anime-sr-warmup")
        self._sr_warmup_pending: set[str] = set()
        self._sr_warmup_lock = RLock()
        self._final_frame_lock = RLock()
        self._final_frame_cache: dict[str, bytes] = {}
        self._final_frame_order: list[str] = []
        self._final_frame_cache_limit = max(32, int(os.environ.get("FLEXAVATAR_ANIME_FINAL_FRAME_CACHE_LIMIT", "320")))
        self._final_frame_disk_cache_enabled = os.environ.get("FLEXAVATAR_ANIME_DISK_FRAME_CACHE", "1") == "1"
        self._final_frame_disk_cache_dir = Path(os.environ.get(
            "FLEXAVATAR_ANIME_FINAL_FRAME_CACHE_DIR",
            self.assets_dir / "sr_webp_cache",
        ))
        if self._final_frame_disk_cache_enabled:
            self._final_frame_disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self._sr_warmup_progress = {
            "active": False,
            "avatar": None,
            "label": "Idle",
            "processedFrames": 0,
            "totalFrames": 0,
            "currentFrame": 0,
            "startedAt": None,
            "completedAt": None,
        }

    @property
    def installed(self):
        return (self.repo_path / "readme.md").exists() or (self.repo_path / "README.md").exists()

    @property
    def expression_handler_installed(self):
        return all(
            (self.tha4_repo_path / "data" / "tha4" / name).exists()
            for name in (
                "eyebrow_decomposer.pt",
                "eyebrow_morphing_combiner.pt",
                "face_morpher.pt",
                "body_morpher.pt",
                "upscaler.pt",
            )
        )

    @property
    def super_resolution_installed(self):
        return self.adore_model_path.exists() or self.realcugan_installed or self.sr_model_path.exists()

    @property
    def realcugan_installed(self):
        return self.realcugan_executable.exists() and self.realcugan_model_dir.exists()

    @property
    def super_resolution_name(self):
        backend = os.environ.get("FLEXAVATAR_ANIME_SR_BACKEND", "real-cugan")
        if backend == "real-cugan" and self.realcugan_installed:
            return "real-cugan"
        if backend == "adore" and self.adore_model_path.exists():
            return "adore"
        if backend == "real-esrgan" and self.sr_model_path.exists():
            return "real-esrgan"
        if self.realcugan_installed:
            return "real-cugan"
        if self.adore_model_path.exists():
            return "adore"
        if self.sr_model_path.exists():
            return "real-esrgan"
        return "off"

    def route_reason(self, image_path: str):
        suffix = Path(image_path).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png"}:
            return None
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        try:
            faces = detect_anime_faces(image)
        except Exception as exc:
            print(f"PAniC anime route detection skipped: {exc}")
            return None
        if faces:
            _, _, x, y, w, h = faces[0]
            return {
                "reason": "anime-face-cascade",
                "bbox": (int(x), int(y), int(w), int(h)),
            }
        return None

    def create_asset(self, avatar_name: str, source_path: str, route: Optional[dict]):
        source = Path(source_path)
        original = Image.open(source)
        image = original.convert("RGBA")
        image = ImageOps.contain(image, (1024, 1024), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (1024, 1024), (255, 255, 255, 0))
        offset = ((1024 - image.width) // 2, (1024 - image.height) // 2)
        canvas.alpha_composite(image, offset)

        bbox = route.get("bbox") if route else None
        if bbox:
            scale = min(1024 / max(1, original.width), 1024 / max(1, original.height))
            x, y, w, h = bbox
            bbox = (
                int(x * scale + offset[0]),
                int(y * scale + offset[1]),
                int(w * scale),
                int(h * scale),
            )

        image_path = self.assets_dir / f"{avatar_name}.png"
        tha4_image_path = self.assets_dir / f"{avatar_name}_tha4.png"
        canvas.save(image_path)
        self._prepare_tha4_source(canvas, bbox).save(tha4_image_path)
        asset = PanicAnimeAsset(
            name=avatar_name,
            source_path=source,
            image_path=image_path,
            tha4_image_path=tha4_image_path,
            face_bbox=bbox,
            image_offset=offset,
            image_size=(image.width, image.height),
        )
        self.queue_slider_cache_warmup(asset)
        return asset

    def render_jpeg(
        self,
        asset: PanicAnimeAsset,
        camera: WebCamera,
        controls: dict,
        frame_index: int,
        width: int,
        height: int,
        quality: int,
        output_format: str = "JPEG",
    ):
        start = time.time()
        expression = controls.get("expression") or []
        head = controls.get("head") or [0, 0, 0]
        mode = controls.get("mode", "default")
        playing = bool(controls.get("playing", True))
        animation_frame = int(time.time() * 30) % 360 if mode == "default" and playing else frame_index

        wave = np.sin(animation_frame / 18) if mode == "default" and playing else 0.0
        blink = max(0.0, np.sin(animation_frame / 7) - 0.82) * 5.5 if mode == "default" and playing else 0.0
        driver_expression = self._driver_expression(expression, mode, wave, blink)
        yaw = camera.yaw + float(head[1] if len(head) > 1 else 0) + wave * 9
        pitch = camera.pitch + float(head[0] if len(head) > 0 else 0) + wave * 2
        roll = camera.roll + float(head[2] if len(head) > 2 else 0) + wave * 2.5
        final_cache_key = None
        should_super_resolve = self._should_super_resolve(mode, playing)
        super_resolved = not should_super_resolve
        if self.expression_handler_installed:
            sr_key = self._sr_pose_cache_key(asset, driver_expression, yaw, pitch, roll, wave)
            if should_super_resolve:
                final_cache_key = self._final_frame_cache_key(
                    asset,
                    driver_expression,
                    yaw,
                    pitch,
                    roll,
                    wave,
                    width,
                    height,
                    camera.radius,
                    output_format,
                    quality,
                )
                cached_frame = self._get_final_frame(final_cache_key)
                if cached_frame is not None:
                    return cached_frame, int(1 / max(time.time() - start, 1e-6))
            base = self._cached_super_resolve_by_key(sr_key) if should_super_resolve else None
            super_resolved = base is not None if should_super_resolve else True
            if base is None:
                base = self._render_tha4_expression(asset, driver_expression, yaw, pitch, roll, wave)
                base = self._crop_to_alpha(base, padding=26)
            if should_super_resolve and not super_resolved:
                base, super_resolved = self._super_resolve(base, sr_key)
            yaw = 0.0
            pitch = 0.0
            roll = 0.0
        else:
            base = Image.open(asset.image_path).convert("RGBA")
            base = self._deform_source(base, asset, driver_expression, wave, yaw, pitch)
            base = self._pose_card(base, yaw, pitch)

        radius_min = 0.35 if self.expression_handler_installed else 0.55
        radius = max(radius_min, min(1.8, camera.radius))
        scale = (0.86 if self.expression_handler_installed else 0.72) / radius
        max_fill = 0.98 if self.expression_handler_installed else 0.92
        target_w = max(96, int(width * min(max_fill, scale)))
        target_h = max(96, int(height * min(max_fill, scale)))
        image = ImageOps.contain(base, (target_w, target_h), Image.Resampling.LANCZOS)
        if self.expression_handler_installed:
            image = self._enhance_anime_display(image)
        image = image.rotate(roll * 0.32, resample=Image.Resampling.BICUBIC, expand=True)

        canvas = Image.new("RGBA", (width, height), (248, 252, 252, 255))
        x = int((width - image.width) / 2 + self._clamp(yaw / 40, -1.0, 1.0) * width * 0.035)
        y = int((height - image.height) / 2 - self._clamp(pitch / 40, -1.0, 1.0) * height * 0.028)
        canvas.alpha_composite(image, (x, y))

        payload = self._encode_frame(canvas, output_format, quality)
        if final_cache_key is not None and super_resolved:
            self._store_final_frame(final_cache_key, payload, output_format)
        return payload, int(1 / max(time.time() - start, 1e-6))

    def _compose_final_frame(
        self,
        base: Image.Image,
        width: int,
        height: int,
        radius: float,
        output_format: str,
        quality: int,
    ):
        radius = max(0.35, min(1.8, radius))
        scale = 0.86 / radius
        target_w = max(96, int(width * min(0.98, scale)))
        target_h = max(96, int(height * min(0.98, scale)))
        image = ImageOps.contain(base, (target_w, target_h), Image.Resampling.LANCZOS)
        image = self._enhance_anime_display(image)
        canvas = Image.new("RGBA", (width, height), (248, 252, 252, 255))
        canvas.alpha_composite(image, ((width - image.width) // 2, (height - image.height) // 2))
        return self._encode_frame(canvas, output_format, quality)

    def _encode_frame(self, canvas: Image.Image, output_format: str, quality: int):
        image_format = output_format.upper()
        buffer = io.BytesIO()
        if image_format == "PNG":
            canvas.save(buffer, format="PNG", optimize=False, compress_level=1)
        elif image_format == "WEBP":
            canvas.convert("RGB").save(
                buffer,
                format="WEBP",
                quality=quality,
                method=int(os.environ.get("FLEXAVATAR_ANIME_WEBP_METHOD", "1")),
                lossless=os.environ.get("FLEXAVATAR_ANIME_WEBP_LOSSLESS", "0") == "1",
            )
        else:
            canvas.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=False)
        return buffer.getvalue()

    def _final_frame_cache_key(
        self,
        asset: PanicAnimeAsset,
        expression: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        wave: float,
        width: int,
        height: int,
        radius: float,
        output_format: str,
        quality: int,
    ):
        digest = hashlib.sha1()
        digest.update(asset.name.encode("utf-8"))
        digest.update(str(asset.tha4_image_path).encode("utf-8"))
        digest.update(str(asset.tha4_image_path.stat().st_mtime_ns).encode("ascii"))
        digest.update(self.super_resolution_name.encode("utf-8"))
        digest.update(str(self.adore_model_path).encode("utf-8"))
        digest.update(str(self.sr_model_path).encode("utf-8"))
        digest.update(str(self.realcugan_model_dir).encode("utf-8"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_SR_INPUT", "").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_SR_MULTIPLE", "8").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_SR_BLEND", "0.68").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_SCALE", "4").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_DENOISE", "-1").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_TTA", "1").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_TILE", "0").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_SYNCGAP", "3").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_JOBS", "1:2:2").encode("ascii"))
        digest.update(str([round(float(value), 3) for value in expression[:32]]).encode("ascii"))
        digest.update(f"{yaw:.3f}:{pitch:.3f}:{roll:.3f}:{wave:.3f}".encode("ascii"))
        digest.update(f"{width}:{height}:{radius:.3f}:{output_format.upper()}:{quality}".encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_WEBP_METHOD", "1").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_WEBP_LOSSLESS", "0").encode("ascii"))
        return digest.hexdigest()

    def _get_final_frame(self, cache_key: str):
        with self._final_frame_lock:
            cached = self._final_frame_cache.get(cache_key)
        if cached is not None:
            return cached
        disk_path = self._final_frame_disk_cache_path(cache_key, "WEBP")
        if disk_path is not None and disk_path.exists():
            try:
                payload = disk_path.read_bytes()
                self._store_final_frame(cache_key, payload, "WEBP", write_disk=False)
                return payload
            except OSError:
                return None
        return None

    def _store_final_frame(self, cache_key: str, payload: bytes, output_format: str = "WEBP", write_disk: bool = True):
        with self._final_frame_lock:
            if cache_key not in self._final_frame_cache:
                self._final_frame_order.append(cache_key)
            self._final_frame_cache[cache_key] = payload
            while len(self._final_frame_order) > self._final_frame_cache_limit:
                old_key = self._final_frame_order.pop(0)
                self._final_frame_cache.pop(old_key, None)
        disk_path = self._final_frame_disk_cache_path(cache_key, output_format)
        if write_disk and disk_path is not None:
            try:
                disk_path.write_bytes(payload)
            except OSError:
                traceback.print_exc()

    def _final_frame_disk_cache_path(self, cache_key: str, output_format: str):
        if not self._final_frame_disk_cache_enabled or output_format.upper() != "WEBP":
            return None
        return self._final_frame_disk_cache_dir / f"{cache_key}.webp"

    def _render_tha4_expression(
        self,
        asset: PanicAnimeAsset,
        expression: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        wave: float,
        warmup: bool = False,
    ):
        handler = self._get_tha4_handler(warmup=warmup)
        return handler.render(asset.tha4_image_path, expression, yaw, pitch, roll, wave)

    def _get_tha4_handler(self, warmup: bool = False):
        if warmup:
            if self._tha4_warmup_handler is None:
                self._tha4_warmup_handler = Tha4ExpressionHandler(self.tha4_repo_path)
            return self._tha4_warmup_handler
        if self._tha4_live_handler is None:
            self._tha4_live_handler = Tha4ExpressionHandler(self.tha4_repo_path)
        return self._tha4_live_handler

    def _should_super_resolve(self, mode: str, playing: bool):
        if not self.super_resolution_installed:
            return False
        if os.environ.get("FLEXAVATAR_DISABLE_ANIME_SR") == "1":
            return False
        return mode == "manual" or not playing

    def _super_resolve(self, image: Image.Image, cache_key: Optional[str] = None):
        handler = self._get_sr_handler()
        return handler.upscale_cached(image, cache_key=cache_key)

    def _cached_super_resolve(self, image: Image.Image, cache_key: Optional[str] = None):
        handler = self._get_sr_handler()
        return handler.cached(image, cache_key=cache_key)

    def _cached_super_resolve_by_key(self, cache_key: str):
        if self._sr_handler is None:
            return None
        return self._sr_handler.cached_by_key(cache_key)

    def _get_sr_handler(self):
        if self._sr_handler is None:
            cache_limit = int(os.environ.get("FLEXAVATAR_ANIME_SR_CACHE_LIMIT", "256"))
            backend = os.environ.get("FLEXAVATAR_ANIME_SR_BACKEND", "real-cugan")
            if (backend == "real-cugan" and self.realcugan_installed) or (
                backend not in {"adore", "real-esrgan"} and self.realcugan_installed
            ):
                max_side = int(os.environ.get("FLEXAVATAR_ANIME_SR_INPUT", "512"))
                scale = int(os.environ.get("FLEXAVATAR_REALCUGAN_SCALE", "4"))
                denoise = int(os.environ.get("FLEXAVATAR_REALCUGAN_DENOISE", "-1"))
                tta = os.environ.get("FLEXAVATAR_REALCUGAN_TTA", "1") == "1"
                tile_size = int(os.environ.get("FLEXAVATAR_REALCUGAN_TILE", "0"))
                syncgap = int(os.environ.get("FLEXAVATAR_REALCUGAN_SYNCGAP", "3"))
                jobs = os.environ.get("FLEXAVATAR_REALCUGAN_JOBS", "1:2:2")
                self._sr_handler = RealCuganSuperResolution(
                    self.realcugan_executable,
                    self.realcugan_model_dir,
                    max_input_side=max_side,
                    scale=scale,
                    denoise=denoise,
                    tta=tta,
                    tile_size=tile_size,
                    syncgap=syncgap,
                    jobs=jobs,
                    cache_limit=cache_limit,
                )
            elif backend == "adore" and self.adore_model_path.exists():
                max_side = int(os.environ.get("FLEXAVATAR_ANIME_SR_INPUT", "512"))
                self._sr_handler = AnimeSuperResolution(self.adore_model_path, max_input_side=max_side, cache_limit=cache_limit)
            elif self.realcugan_installed:
                max_side = int(os.environ.get("FLEXAVATAR_ANIME_SR_INPUT", "512"))
                scale = int(os.environ.get("FLEXAVATAR_REALCUGAN_SCALE", "4"))
                denoise = int(os.environ.get("FLEXAVATAR_REALCUGAN_DENOISE", "-1"))
                tta = os.environ.get("FLEXAVATAR_REALCUGAN_TTA", "1") == "1"
                tile_size = int(os.environ.get("FLEXAVATAR_REALCUGAN_TILE", "0"))
                syncgap = int(os.environ.get("FLEXAVATAR_REALCUGAN_SYNCGAP", "3"))
                jobs = os.environ.get("FLEXAVATAR_REALCUGAN_JOBS", "1:2:2")
                self._sr_handler = RealCuganSuperResolution(
                    self.realcugan_executable,
                    self.realcugan_model_dir,
                    max_input_side=max_side,
                    scale=scale,
                    denoise=denoise,
                    tta=tta,
                    tile_size=tile_size,
                    syncgap=syncgap,
                    jobs=jobs,
                    cache_limit=cache_limit,
                )
            elif self.adore_model_path.exists():
                max_side = int(os.environ.get("FLEXAVATAR_ANIME_SR_INPUT", "512"))
                self._sr_handler = AnimeSuperResolution(self.adore_model_path, max_input_side=max_side, cache_limit=cache_limit)
            else:
                max_side = int(os.environ.get("FLEXAVATAR_ANIME_SR_INPUT", "256"))
                self._sr_handler = AnimeSuperResolution(self.sr_model_path, max_input_side=max_side, cache_limit=cache_limit)
        return self._sr_handler

    def queue_slider_cache_warmup(self, asset: PanicAnimeAsset):
        if os.environ.get("FLEXAVATAR_DISABLE_ANIME_SR_PREWARM") == "1":
            return
        if not self.super_resolution_installed or not self.expression_handler_installed:
            return
        warmup_key = f"{asset.name}:{self.super_resolution_name}:{asset.tha4_image_path.stat().st_mtime_ns}"
        poses = list(self._slider_warmup_poses())
        with self._sr_warmup_lock:
            if warmup_key in self._sr_warmup_pending:
                return
            self._sr_warmup_pending.add(warmup_key)
            self._sr_warmup_progress = {
                "active": True,
                "avatar": asset.name,
                "label": "Preprocessing slider SR frames",
                "processedFrames": 0,
                "totalFrames": len(poses),
                "currentFrame": 0,
                "startedAt": time.time(),
                "completedAt": None,
            }
        with self._final_frame_lock:
            self._final_frame_cache.clear()
            self._final_frame_order.clear()
        self._sr_warmup_executor.submit(self._warm_slider_cache_worker, warmup_key, asset, poses)

    def warm_current_pose(self, asset: PanicAnimeAsset, camera: WebCamera, controls: dict, frame_index: int):
        if os.environ.get("FLEXAVATAR_DISABLE_ANIME_SR_PREWARM") == "1":
            return
        if not self._should_super_resolve(str(controls.get("mode", "default")), bool(controls.get("playing", True))):
            return
        expression = controls.get("expression") or []
        head = controls.get("head") or [0, 0, 0]
        driver_expression = self._driver_expression(expression, "manual", 0.0, 0.0)
        yaw = camera.yaw + float(head[1] if len(head) > 1 else 0)
        pitch = camera.pitch + float(head[0] if len(head) > 0 else 0)
        roll = camera.roll + float(head[2] if len(head) > 2 else 0)
        pose_key = self._warmup_pose_key(asset, driver_expression, yaw, pitch, roll, frame_index)
        with self._sr_warmup_lock:
            if pose_key in self._sr_warmup_pending:
                return
            self._sr_warmup_pending.add(pose_key)
        self._sr_warmup_executor.submit(
            self._warm_single_pose_worker,
            pose_key,
            asset,
            driver_expression,
            yaw,
            pitch,
            roll,
            0.0,
        )

    def sr_stats(self):
        if self._sr_handler is None:
            stats = {"cached": 0, "pending": 0, "limit": 0}
        else:
            stats = self._sr_handler.stats()
        stats["warmup"] = self._warmup_progress()
        with self._final_frame_lock:
            stats["finalFrames"] = len(self._final_frame_cache)
            stats["finalFrameLimit"] = self._final_frame_cache_limit
        return stats

    def _warm_slider_cache_worker(self, warmup_key: str, asset: PanicAnimeAsset, poses: list[tuple[list[float], float, float, float]]):
        try:
            for frame_number, (expression, yaw, pitch, roll) in enumerate(poses, start=1):
                if os.environ.get("FLEXAVATAR_DISABLE_ANIME_SR_PREWARM") == "1":
                    break
                with self._sr_warmup_lock:
                    if self._sr_warmup_progress.get("avatar") == asset.name:
                        self._sr_warmup_progress["currentFrame"] = frame_number
                self._warm_single_pose(asset, expression, yaw, pitch, roll, 0.0)
                with self._sr_warmup_lock:
                    if self._sr_warmup_progress.get("avatar") == asset.name:
                        self._sr_warmup_progress["processedFrames"] = frame_number
        except Exception:
            traceback.print_exc()
        finally:
            with self._sr_warmup_lock:
                self._sr_warmup_pending.discard(warmup_key)
                if self._sr_warmup_progress.get("avatar") == asset.name:
                    total = int(self._sr_warmup_progress.get("totalFrames") or 0)
                    processed = int(self._sr_warmup_progress.get("processedFrames") or 0)
                    self._sr_warmup_progress["active"] = False
                    self._sr_warmup_progress["currentFrame"] = processed
                    self._sr_warmup_progress["completedAt"] = time.time()
                    self._sr_warmup_progress["label"] = (
                        "SR prepass complete" if processed >= total else "SR prepass stopped"
                    )

    def _warm_single_pose_worker(
        self,
        pose_key: str,
        asset: PanicAnimeAsset,
        expression: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        wave: float,
    ):
        try:
            self._warm_single_pose(asset, expression, yaw, pitch, roll, wave)
        except Exception:
            traceback.print_exc()
        finally:
            with self._sr_warmup_lock:
                self._sr_warmup_pending.discard(pose_key)

    def _warm_single_pose(
        self,
        asset: PanicAnimeAsset,
        expression: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        wave: float,
    ):
        image = self._render_tha4_expression(asset, expression, yaw, pitch, roll, wave, warmup=True)
        image = self._crop_to_alpha(image, padding=26)
        cache_key = self._sr_pose_cache_key(asset, expression, yaw, pitch, roll, wave)
        self._get_sr_handler().warm(image, cache_key=cache_key)
        sr_image = self._cached_super_resolve_by_key(cache_key)
        if sr_image is not None:
            width = int(os.environ.get("FLEXAVATAR_ANIME_STILL_WIDTH", "1600"))
            height = int(os.environ.get("FLEXAVATAR_ANIME_STILL_HEIGHT", "1600"))
            quality = int(os.environ.get("FLEXAVATAR_ANIME_WEBP_QUALITY", "96"))
            radius = float(os.environ.get("FLEXAVATAR_ANIME_STILL_RADIUS", "1.0"))
            final_key = self._final_frame_cache_key(
                asset,
                expression,
                yaw,
                pitch,
                roll,
                wave,
                width,
                height,
                radius,
                "WEBP",
                quality,
            )
            if self._get_final_frame(final_key) is None:
                payload = self._compose_final_frame(sr_image, width, height, radius, "WEBP", quality)
                self._store_final_frame(final_key, payload, "WEBP")

    def _sr_pose_cache_key(
        self,
        asset: PanicAnimeAsset,
        expression: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        wave: float,
    ):
        digest = hashlib.sha1()
        digest.update(asset.name.encode("utf-8"))
        digest.update(str(asset.tha4_image_path).encode("utf-8"))
        digest.update(str(asset.tha4_image_path.stat().st_mtime_ns).encode("ascii"))
        digest.update(self.super_resolution_name.encode("utf-8"))
        digest.update(str(self.adore_model_path).encode("utf-8"))
        digest.update(str(self.sr_model_path).encode("utf-8"))
        digest.update(str(self.realcugan_model_dir).encode("utf-8"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_SR_INPUT", "").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_SR_MULTIPLE", "8").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_ANIME_SR_BLEND", "0.68").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_SCALE", "4").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_DENOISE", "-1").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_TTA", "1").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_TILE", "0").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_SYNCGAP", "3").encode("ascii"))
        digest.update(os.environ.get("FLEXAVATAR_REALCUGAN_JOBS", "1:2:2").encode("ascii"))
        digest.update(str([round(float(value), 3) for value in expression[:32]]).encode("ascii"))
        digest.update(f"{yaw:.3f}:{pitch:.3f}:{roll:.3f}:{wave:.3f}".encode("ascii"))
        return digest.hexdigest()

    def _warmup_progress(self):
        with self._sr_warmup_lock:
            progress = dict(self._sr_warmup_progress)
            progress["queuedWarmups"] = len(self._sr_warmup_pending)
        total = int(progress.get("totalFrames") or 0)
        processed = int(progress.get("processedFrames") or 0)
        progress["remainingFrames"] = max(0, total - processed)
        progress["percent"] = round((processed / total) * 100, 1) if total else 0.0
        return progress

    def _slider_warmup_poses(self):
        neutral = [0.0] * 32
        yield neutral, 0.0, 0.0, 0.0
        control_ranges = {
            0: (-1.0, 2.0),
            9: (0.0, 1.5),
            17: (-1.0, 1.0),
            18: (-1.0, 1.0),
        }
        for index in range(21):
            minimum, maximum = control_ranges.get(index, (0.0, 2.0))
            values = self._warmup_values(minimum, maximum, step=0.25)
            for value in values:
                expression = [0.0] * 32
                expression[index] = value
                yield self._driver_expression(expression, "manual", 0.0, 0.0), 0.0, 0.0, 0.0
        for yaw in self._warmup_values(-32.0, 32.0, step=4.0):
            yield neutral, yaw, 0.0, 0.0
        for pitch in self._warmup_values(-24.0, 24.0, step=4.0):
            yield neutral, 0.0, pitch, 0.0
        for roll in self._warmup_values(-18.0, 18.0, step=3.0):
            yield neutral, 0.0, 0.0, roll

    @staticmethod
    def _warmup_values(minimum: float, maximum: float, step: float = 0.25):
        values = []
        current = minimum
        while current <= maximum + 1e-6:
            rounded = round(float(current), 2)
            if abs(rounded) > 1e-6:
                values.append(rounded)
            current += step
        return values

    def _warmup_pose_key(
        self,
        asset: PanicAnimeAsset,
        expression: list[float],
        yaw: float,
        pitch: float,
        roll: float,
        frame_index: int,
    ):
        digest = hashlib.sha1()
        digest.update(asset.name.encode("utf-8"))
        digest.update(self.super_resolution_name.encode("utf-8"))
        digest.update(str([round(float(value), 2) for value in expression[:32]]).encode("ascii"))
        digest.update(f"{yaw:.2f}:{pitch:.2f}:{roll:.2f}".encode("ascii"))
        return digest.hexdigest()

    def _driver_expression(self, expression: list[float], mode: str, wave: float, blink: float):
        values = [0.0] * 32
        for index, value in enumerate(expression[:32]):
            values[index] = float(value)
        if mode == "default":
            values[0] += 0.35 + 0.22 * wave
            values[10] += blink
        return values

    def _deform_source(
        self,
        base: Image.Image,
        asset: PanicAnimeAsset,
        expression: list[float],
        wave: float,
        yaw: float,
        pitch: float,
    ):
        rgba = np.array(base)
        if rgba.shape[2] != 4:
            return base

        h, w = rgba.shape[:2]
        bbox = asset.face_bbox or self._fallback_face_box(asset)
        if bbox is None:
            return base

        face_x, face_y, face_w, face_h = bbox
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = np.zeros((h, w), dtype=np.float32)
        dy = np.zeros((h, w), dtype=np.float32)

        brow = self._clamp(expression[0] / 2.0, -1.0, 1.0)
        squint = self._clamp(expression[7] / 2.0, -1.0, 1.0)
        yaw_norm = self._clamp(yaw / 35.0, -1.0, 1.0)
        pitch_norm = self._clamp(pitch / 35.0, -1.0, 1.0)

        cx = face_x + face_w * 0.5
        eye_y = face_y + face_h * 0.39
        brow_y = face_y + face_h * 0.27

        eye_weight = self._gaussian_2d(xx, yy, cx, eye_y, face_w * 0.48, face_h * 0.105)
        brow_weight = self._gaussian_2d(xx, yy, cx, brow_y, face_w * 0.45, face_h * 0.095)

        dy += brow_weight * brow * face_h * 0.025
        dy += eye_weight * np.sign(yy - eye_y) * squint * face_h * 0.035
        dx += (brow_weight + eye_weight) * yaw_norm * face_w * 0.022
        dy -= (brow_weight + eye_weight) * pitch_norm * face_h * 0.022

        if abs(wave) > 0.001:
            breathing = self._gaussian_2d(xx, yy, cx, face_y + face_h * 1.28, face_w * 0.75, face_h * 0.42)
            dy += breathing * wave * face_h * 0.018

        map_x = np.clip(xx - dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(yy - dy, 0, h - 1).astype(np.float32)
        warped = cv2.remap(rgba, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        return Image.fromarray(warped, mode="RGBA")

    def _pose_card(self, image: Image.Image, yaw: float, pitch: float):
        yaw_norm = self._clamp(yaw / 40.0, -1.0, 1.0)
        pitch_norm = self._clamp(pitch / 40.0, -1.0, 1.0)
        if abs(yaw_norm) < 0.01 and abs(pitch_norm) < 0.01:
            return image

        rgba = np.array(image)
        h, w = rgba.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        yaw_mag = abs(yaw_norm)
        pitch_mag = abs(pitch_norm)
        left_inset = yaw_mag * w * (0.018 if yaw_norm > 0 else 0.085)
        right_inset = yaw_mag * w * (0.085 if yaw_norm > 0 else 0.018)
        top_shift = pitch_norm * h * 0.032
        bottom_shift = -pitch_norm * h * 0.026
        top_squeeze = pitch_mag * h * 0.025
        bottom_squeeze = pitch_mag * h * 0.018
        dst = np.float32([
            [left_inset, top_squeeze + top_shift],
            [w - right_inset, top_squeeze - top_shift],
            [w - right_inset * 0.55, h - bottom_squeeze - bottom_shift],
            [left_inset * 0.55, h - bottom_squeeze + bottom_shift],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            rgba,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255, 0),
        )
        return Image.fromarray(warped, mode="RGBA")

    def _fallback_face_box(self, asset: PanicAnimeAsset):
        offset_x, offset_y = asset.image_offset
        image_w, image_h = asset.image_size
        if image_w <= 0 or image_h <= 0:
            return None
        side = min(image_w, image_h)
        return (
            int(offset_x + image_w * 0.25),
            int(offset_y + image_h * 0.14),
            int(side * 0.50),
            int(side * 0.52),
        )

    def _prepare_tha4_source(self, canvas: Image.Image, bbox: Optional[tuple[int, int, int, int]]):
        target_size = 512
        target_face = (192.0, 64.0, 128.0, 128.0)
        source = self._make_border_background_transparent(canvas)
        if bbox is None:
            return ImageOps.contain(source, (target_size, target_size), Image.Resampling.LANCZOS)

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return ImageOps.contain(source, (target_size, target_size), Image.Resampling.LANCZOS)

        face_cx = x + w * 0.5
        face_cy = y + h * 0.5
        target_face_cx = target_face[0] + target_face[2] * 0.5
        target_face_cy = target_face[1] + target_face[3] * 0.5
        scale = target_face[2] / max(float(w), 1.0)
        crop_side = target_size / max(scale, 1e-6)
        left = face_cx - target_face_cx / scale
        top = face_cy - target_face_cy / scale
        return self._crop_with_transparent_padding(source, left, top, crop_side).resize(
            (target_size, target_size),
            Image.Resampling.LANCZOS,
        )

    @staticmethod
    def _make_border_background_transparent(image: Image.Image):
        rgba = np.array(image.convert("RGBA"))
        rgb = rgba[:, :, :3].astype(np.int16)
        alpha = rgba[:, :, 3]
        near_white = np.all(rgb > 238, axis=2) & (np.max(rgb, axis=2) - np.min(rgb, axis=2) < 14)
        h, w = near_white.shape
        _, labels = cv2.connectedComponents(near_white.astype(np.uint8))
        transparent = np.zeros_like(near_white)
        border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
        for label in border_labels:
            if label != 0:
                transparent |= labels == label
        alpha[transparent] = 0
        rgba[:, :, 3] = alpha
        return Image.fromarray(rgba)

    @staticmethod
    def _crop_with_transparent_padding(image: Image.Image, left: float, top: float, side: float):
        left_i = int(np.floor(left))
        top_i = int(np.floor(top))
        right_i = int(np.ceil(left + side))
        bottom_i = int(np.ceil(top + side))
        output = Image.new("RGBA", (max(1, right_i - left_i), max(1, bottom_i - top_i)), (255, 255, 255, 0))
        crop_left = max(0, left_i)
        crop_top = max(0, top_i)
        crop_right = min(image.width, right_i)
        crop_bottom = min(image.height, bottom_i)
        if crop_right > crop_left and crop_bottom > crop_top:
            crop = image.crop((crop_left, crop_top, crop_right, crop_bottom))
            output.alpha_composite(crop, (crop_left - left_i, crop_top - top_i))
        return output

    @staticmethod
    def _crop_to_alpha(image: Image.Image, padding: int = 0):
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        bbox = alpha.getbbox()
        if bbox is None:
            return rgba
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(rgba.width, bbox[2] + padding)
        bottom = min(rgba.height, bbox[3] + padding)
        return rgba.crop((left, top, right, bottom))

    @staticmethod
    def _enhance_anime_display(image: Image.Image):
        rgba = image.convert("RGBA")
        alpha = rgba.getchannel("A")
        rgb = rgba.convert("RGB")
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=0.85, percent=82, threshold=3))
        rgb = rgb.filter(ImageFilter.SHARPEN)
        enhanced = Image.merge("RGBA", (*rgb.split(), alpha))
        return enhanced

    @staticmethod
    def _gaussian_2d(xx, yy, cx: float, cy: float, sx: float, sy: float):
        sx = max(float(sx), 1.0)
        sy = max(float(sy), 1.0)
        return np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy))).astype(np.float32)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float):
        return max(minimum, min(maximum, float(value)))


class WebAvatarSession:
    def __init__(self):
        self.lock = RLock()
        self.status = "Booting FlexAvatar web runtime..."
        self.current_avatar = "marble_sculpture"
        self.renderer = "flexavatar"
        self.mode = "default"
        self.playing = True
        self.lock_head = False
        self.busy = False
        self.last_error = None
        self.last_generated_avatar = None
        self.frame_index = 0
        self.render_fps = 0
        self.animation_fps = 0
        self.webcam_ready = False
        self.export_dir = Path(REPO_ROOT) / "data" / "web_exports"
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self._panic_backend = PanicAnimeBackend()
        self._panic_asset: PanicAnimeAsset | None = None
        self._driver_controls = {
            "mode": self.mode,
            "playing": self.playing,
            "lockHead": self.lock_head,
            "expression": [0.0] * 32,
            "jaw": [0.0, 0.0, 0.0],
            "head": [0.0, 0.0, 0.0],
        }

        self.camera = WebCamera()
        self.gaussians = GaussianModel(3)
        self._window_closed = False
        self._sheap_module = None
        self._panic_driver_prev_gray = None
        self._last_panic_jpeg = None

        self._load_runtime()
        self._animation_thread = Thread(target=self._animate_avatar, daemon=True)
        self._animation_thread.start()

    def _set_status(self, status: str):
        self.status = status
        print(status)

    def prepare_generation(self):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
            self.last_error = None
            self.last_generated_avatar = None
            self.status = "Generation queued."
        print(self.status)
        return True

    def _load_runtime(self):
        self._set_status("Loading FlexAvatar model...")
        device = torch.device("cuda")
        model_manager = FlexAvatarModelManager("FLEX-1")
        self._model = model_manager.load_checkpoint(-1).to(device)
        self._avatar_code_manager = AvatarCodeManager()
        self._dataset_config = model_manager.load_dataset_config()
        self._preprocessor = FlexAvatarPreprocessor(self._dataset_config)

        self._set_status("Loading default avatar and animation...")
        self._load_avatar_for_name(self.current_avatar)

        data_adapter_driver = NeRSembleDataAdapter(
            240,
            "SEN-10-port_strong_smokey",
            expression_code_config=self._dataset_config.expression_code_config,
        )
        timesteps = data_adapter_driver.list_timesteps()
        expression_codes = [
            data_adapter_driver.load_expression_code(SampleMetadata(None, None, timestep, None))
            for timestep in timesteps
        ]
        expression_codes.extend(interpolate_codes([expression_codes[-1], expression_codes[0]], n_frames=12))
        self._expression_codes = [torch.tensor(code, dtype=torch.float32, device=device) for code in expression_codes]
        self._last_expression_code = self._expression_codes[0]
        self._manual_expression_code = self._make_neutral_expression_code(device)
        self._set_status("Ready. Start with an image, generate, preview, then use webcam.")

    def _make_neutral_expression_code(self, device):
        neutral = torch.zeros(135, dtype=torch.float32, device=device)
        identity_6d = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32, device=device)
        neutral[100:106] = identity_6d
        neutral[106:112] = identity_6d
        neutral[114:120] = identity_6d
        neutral[120:126] = identity_6d
        neutral[126:132] = identity_6d
        return neutral

    @staticmethod
    def avatar_name_from_path(path: str):
        stem = Path(path).stem
        if stem.startswith("avatar_code_"):
            stem = stem[len("avatar_code_"):]
        cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)
        return cleaned.strip("_") or "avatar"

    def _copy_avatar_input(self, image_path: str):
        source = Path(image_path)
        suffix = source.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".mp4"}:
            raise ValueError(f"Unsupported input type: {suffix}")
        avatar_name = self.avatar_name_from_path(str(source))
        dest_suffix = ".jpg" if suffix == ".jpeg" else suffix
        dest = Path(FLEXAVATAR_INPUTS_PATH) / "itw" / f"{avatar_name}{dest_suffix}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != dest.resolve():
            copy2(source, dest)
        return avatar_name, str(dest)

    def _load_avatar_for_name(self, avatar_name: str):
        if not self._avatar_code_manager.has_avatar_code(avatar_name):
            raise FileNotFoundError(self._avatar_code_manager.get_avatar_code_path(avatar_name))

        device = torch.device("cuda")
        data_adapter_source = InTheWildDataAdapter(
            avatar_name,
            expression_code_config=self._dataset_config.expression_code_config,
        )
        batch = create_example_batch(data_adapter_source, avatar_name)
        batch = self._preprocessor.process(batch.to(device))
        avatar_code = self._avatar_code_manager.load_avatar_code(avatar_name).to(device)

        with torch.no_grad():
            output = self._model.create_gaussian_models(
                batch.input_images,
                batch.features,
                batch.input_cam2worlds,
                batch.input_intrinsics,
                expression_codes=batch.input_expression_codes,
                dataset_ids=batch.dataset_ids,
                cached_internal_representations=avatar_code,
            )

        with self.lock:
            self.renderer = "flexavatar"
            self._panic_asset = None
            self.current_avatar = avatar_name
            self._batch = batch
            self._avatar_code = output.internal_representations
            self.gaussians = output.gaussian_models[0][0]

    def _generate_panic_avatar(self, avatar_name: str, copied_path: str, route: Optional[dict]):
        asset = self._panic_backend.create_asset(avatar_name, copied_path, route)
        with self.lock:
            self.renderer = "panic3d"
            self.current_avatar = avatar_name
            self._panic_asset = asset
            self.last_generated_avatar = avatar_name
            self.mode = "default"
            self.playing = True
            self.webcam_ready = False
        if self._panic_backend.installed:
            self._set_status(
                f"Ready: {avatar_name} routed to the PAniC-3D anime backend. "
                "Preview uses the THA4 expression handler while reconstruction export is wired next."
            )
        else:
            self._set_status(
                f"Ready: {avatar_name} routed to the PAniC-3D anime backend contract. "
                "THA4 handles expressions; install PAniC-3D models to replace the preview geometry."
            )

    def generate_avatar(self, input_path: str):
        if not self.busy and not self.prepare_generation():
            return
        try:
            avatar_name, copied_path = self._copy_avatar_input(input_path)
            panic_route = self._panic_backend.route_reason(copied_path)
            if panic_route and os.environ.get("FLEXAVATAR_FORCE_FLAME_FOR_ANIME") != "1":
                self._set_status(f"Detected anime input for {avatar_name}; routing to PAniC-3D backend...")
                self._generate_panic_avatar(avatar_name, copied_path, panic_route)
                return

            self._set_status(f"Tracking {avatar_name} with Pixel3DMM...")
            try:
                run_pixel3dmm(copied_path)
            except Exception as exc:
                if (
                    "Anime face fallback detected" in str(exc)
                    and os.environ.get("FLEXAVATAR_FORCE_FLAME_FOR_ANIME") != "1"
                ):
                    self._set_status(f"Pixel3DMM identified {avatar_name} as anime; routing to PAniC-3D backend...")
                    self._generate_panic_avatar(avatar_name, copied_path, panic_route)
                    return
                raise

            self._set_status(f"Generating avatar code for {avatar_name}...")
            device = torch.device("cuda")
            data_adapter_source = InTheWildDataAdapter(
                avatar_name,
                expression_code_config=self._dataset_config.expression_code_config,
            )
            batch = create_example_batch(data_adapter_source, avatar_name)
            batch = self._preprocessor.process(batch.to(device))

            with torch.no_grad():
                output = self._model.create_gaussian_models(
                    batch.input_images,
                    batch.features,
                    batch.input_cam2worlds,
                    batch.input_intrinsics,
                    expression_codes=batch.input_expression_codes,
                    dataset_ids=batch.dataset_ids,
                )

            self._avatar_code_manager.save_avatar_code(output.internal_representations, avatar_name)
            with self.lock:
                self.renderer = "flexavatar"
                self._panic_asset = None
                self.current_avatar = avatar_name
                self._batch = batch
                self._avatar_code = output.internal_representations
                self.gaussians = output.gaussian_models[0][0]
                self.last_generated_avatar = avatar_name
            self._set_status(f"Ready: {avatar_name}. Preview, tune sliders, then drive with webcam.")
        except Exception as exc:
            traceback.print_exc()
            self.last_error = str(exc)
            self._set_status(f"Generation failed: {exc}")
        finally:
            self.busy = False

    def apply_controls(self, payload: ControlPayload):
        with self.lock:
            self.mode = payload.mode
            self.playing = payload.playing
            self.lock_head = payload.lock_head
            self._driver_controls = {
                "mode": payload.mode,
                "playing": payload.playing,
                "lockHead": payload.lock_head,
                "expression": [float(value) for value in payload.expression[:32]],
                "jaw": [float(value) for value in payload.jaw[:3]],
                "head": [float(value) for value in payload.head[:3]],
            }
            camera = payload.camera or {}
            self.camera.yaw = float(camera.get("yaw", self.camera.yaw))
            self.camera.pitch = float(camera.get("pitch", self.camera.pitch))
            self.camera.roll = float(camera.get("roll", self.camera.roll))
            self.camera.radius = float(camera.get("radius", self.camera.radius))
            if self.renderer == "panic3d":
                self._set_status("PAniC anime driver controls updated.")
                return
            for index, value in enumerate(payload.expression[:100]):
                self._manual_expression_code[index] = float(value)
            self._apply_rotation(payload.jaw[:3], 120)
            self._apply_rotation(payload.head[:3], 126)
            self._set_status("Preview controls updated.")

    def _apply_rotation(self, euler_angles: list[float], start: int):
        if len(euler_angles) < 3:
            return
        rot_matrix = torch.tensor(
            R.from_euler("xyz", euler_angles[:3], degrees=True).as_matrix(),
            dtype=torch.float32,
            device=self._manual_expression_code.device,
        )
        self._manual_expression_code[start:start + 6] = matrix_to_rotation_6d(rot_matrix)

    def _animate_avatar(self):
        last_time = time.time()
        while not self._window_closed:
            if self.busy:
                time.sleep(0.05)
                continue

            is_non_flex_renderer = False
            with self.lock:
                if self.renderer != "flexavatar":
                    if self.playing:
                        self.frame_index = (self.frame_index + 1) % 360
                    now = time.time()
                    self.animation_fps = int(1 / max(now - last_time, 1e-6))
                    last_time = now
                    is_non_flex_renderer = True
                if not is_non_flex_renderer:
                    mode = self.mode
                    playing = self.playing
                    lock_head = self.lock_head
                    batch = self._batch
                    avatar_code = self._avatar_code
                    if mode == "webcam":
                        expression_code = self._last_expression_code
                    elif mode == "manual":
                        expression_code = self._manual_expression_code
                    else:
                        expression_code = self._expression_codes[self.frame_index]
                        if playing:
                            self.frame_index = (self.frame_index + 1) % len(self._expression_codes)
            if is_non_flex_renderer:
                time.sleep(1 / 30)
                continue

            animated_batch = replace(batch, expression_codes=expression_code[None][None])
            with torch.no_grad():
                output = self._model.forward(
                    animated_batch,
                    cached_internal_representations=avatar_code,
                    only_gaussian_models=True,
                )

            gaussians = output.gaussian_models_output.gaussian_models[0][0]
            if not lock_head:
                rotation = rotation_6d_to_matrix(expression_code[-9:-3])
                translation = expression_code[-3:]
                flame2world = torch.eye(4).cuda()
                flame2world[:3, :3] = rotation
                flame2world[:3, 3] = translation
                transform_gaussian_model(gaussians, flame2world)

            now = time.time()
            self.animation_fps = int(1 / max(now - last_time, 1e-6))
            last_time = now
            with self.lock:
                self.gaussians = gaussians

    def render_jpeg(self, width: int = 1280, height: int = 720, quality: int = 90, output_format: str = "JPEG"):
        start = time.time()
        with self.lock:
            self.camera.image_width = width
            self.camera.image_height = height
            if self.renderer == "panic3d" and self._panic_asset is not None:
                asset = self._panic_asset
                camera = replace(self.camera)
                controls = {
                    "mode": self._driver_controls.get("mode"),
                    "playing": self._driver_controls.get("playing"),
                    "lockHead": self._driver_controls.get("lockHead"),
                    "expression": list(self._driver_controls.get("expression") or []),
                    "jaw": list(self._driver_controls.get("jaw") or []),
                    "head": list(self._driver_controls.get("head") or []),
                }
                frame_index = self.frame_index
                last_panic_jpeg = self._last_panic_jpeg
            else:
                asset = None

        if asset is not None:
            try:
                jpeg, fps = self._panic_backend.render_jpeg(
                    asset,
                    camera,
                    controls,
                    frame_index,
                    width,
                    height,
                    quality,
                    output_format,
                )
                with self.lock:
                    self.render_fps = fps
                    self._last_panic_jpeg = jpeg
                return jpeg
            except Exception as exc:
                traceback.print_exc()
                with self.lock:
                    self.status = f"Preview render recovered: {exc}"
                    self.render_fps = 0
                    if self._last_panic_jpeg is not None:
                        return self._last_panic_jpeg
                if last_panic_jpeg is not None:
                    return last_panic_jpeg
                return self._placeholder_jpeg(width, height, quality)

        with self.lock:
            self.camera.image_width = width
            self.camera.image_height = height

            class Cam:
                FoVx = float(np.radians(self.camera.fovx))
                FoVy = float(np.radians(self.camera.fovy))
                image_height = self.camera.image_height
                image_width = self.camera.image_width
                world_view_transform = torch.tensor(self.camera.world_view_transform).float().cuda().T
                full_proj_transform = torch.tensor(self.camera.full_proj_transform).float().cuda().T
                camera_center = torch.tensor(self.camera.pose[:3, 3]).cuda()
                cx = self.camera.image_width / 2
                cy = self.camera.image_height / 2

            rendering_output = render_gsplat(Cam, self.gaussians, torch.tensor((1.0, 1.0, 1.0)).cuda())
            rgb = rendering_output["render"].permute(1, 2, 0).detach().clamp(0, 1).cpu().numpy()

        image = Image.fromarray((rgb * 255).astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        self.render_fps = int(1 / max(time.time() - start, 1e-6))
        return buffer.getvalue()

    @staticmethod
    def _placeholder_jpeg(width: int, height: int, quality: int = 82):
        canvas = Image.new("RGB", (width, height), (248, 252, 252))
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()

    def export_ply(self):
        with self.lock:
            if self.renderer != "flexavatar":
                raise RuntimeError("PLY export is only available for FlexAvatar Gaussian avatars.")
            path = self.export_dir / f"{self.current_avatar}.ply"
            original_scaling = self.gaussians._scaling
            original_opacity = self.gaussians._opacity
            try:
                self.gaussians._scaling = torch.log(original_scaling.clamp_min(1e-8))
                self.gaussians._opacity = torch.logit(original_opacity.clamp(1e-6, 1 - 1e-6))
                self.gaussians.save_ply(str(path))
            finally:
                self.gaussians._scaling = original_scaling
                self.gaussians._opacity = original_opacity
        self._set_status(f"Exported web splat PLY: {path.name}")
        return path

    def drive_from_image(self, image_bytes: bytes):
        with self.lock:
            is_panic = self.renderer == "panic3d"
        if is_panic:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            self._drive_panic_from_frame(np.array(image))
            return
        if self._sheap_module is None:
            self._set_status("Loading SHeaP webcam driver...")
            self._sheap_module = SheapModule()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)
        if frame.shape[1] != frame.shape[0]:
            crop_x = max((frame.shape[1] - frame.shape[0]) // 2, 0)
            frame = frame[:, crop_x: frame.shape[0] + crop_x]
        sheap_output = self._sheap_module(frame)
        expression_code = self._sheap_module.to_expression_code(sheap_output)[0]
        expression_code[-3:] = 0
        with self.lock:
            self._last_expression_code = expression_code
            self.mode = "webcam"
            self.webcam_ready = True
        self._set_status("Driving avatar from browser webcam.")

    def _drive_panic_from_frame(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))

        frame_h, frame_w = gray.shape[:2]
        expression = [0.0] * 32
        head = [0.0, 0.0, 0.0]

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
            face_cx = x + w / 2
            face_cy = y + h / 2
            head[1] = self._clamp((0.5 - face_cx / max(frame_w, 1)) * 34, -18, 18)
            head[0] = self._clamp((face_cy / max(frame_h, 1) - 0.48) * 28, -14, 14)

            mouth_roi = gray[
                int(y + h * 0.62): int(y + h * 0.86),
                int(x + w * 0.32): int(x + w * 0.68),
            ]
            if mouth_roi.size:
                threshold = max(20, float(np.mean(mouth_roi) - np.std(mouth_roi) * 0.55))
                dark_ratio = float(np.mean(mouth_roi < threshold))
                expression[4] = self._clamp((dark_ratio - 0.18) * 7.5, 0.0, 1.8)
                expression[8] = self._clamp((w / max(frame_w, 1) - 0.20) * 3.0, 0.0, 1.0)

            eye_roi = gray[
                int(y + h * 0.26): int(y + h * 0.48),
                int(x + w * 0.18): int(x + w * 0.82),
            ]
            if eye_roi.size:
                expression[10] = self._clamp((120 - float(np.mean(eye_roi))) / 50, 0.0, 1.3)
        elif self._panic_driver_prev_gray is not None:
            prev = cv2.resize(self._panic_driver_prev_gray, (frame_w, frame_h))
            diff = cv2.absdiff(gray, prev)
            moments = cv2.moments(diff)
            energy = float(np.mean(diff))
            if moments["m00"] > 1e-3:
                motion_x = moments["m10"] / moments["m00"]
                motion_y = moments["m01"] / moments["m00"]
                head[1] = self._clamp((0.5 - motion_x / max(frame_w, 1)) * 20, -10, 10)
                head[0] = self._clamp((motion_y / max(frame_h, 1) - 0.5) * 16, -8, 8)
            expression[4] = self._clamp((energy - 4) / 18, 0.0, 1.2)

        self._panic_driver_prev_gray = cv2.resize(gray, (160, 160))
        with self.lock:
            self.mode = "webcam"
            self.playing = False
            self.webcam_ready = True
            self._driver_controls["mode"] = "webcam"
            self._driver_controls["playing"] = False
            self._driver_controls["expression"] = expression
            self._driver_controls["head"] = head
        self._set_status("Driving PAniC anime adapter from browser webcam.")

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float):
        return max(minimum, min(maximum, float(value)))

    def state(self):
        points = 0 if self.renderer != "flexavatar" else int(self.gaussians._xyz.shape[0]) if self.gaussians._xyz is not None else 0
        if self.renderer == "panic3d":
            renderer_label = "PAniC-3D + THA4" if self._panic_backend.installed else "THA4 Anime Expressions"
        else:
            renderer_label = "FlexAvatar Gaussian"
        return {
            "status": self.status,
            "busy": self.busy,
            "currentAvatar": self.current_avatar,
            "renderer": self.renderer,
            "rendererLabel": renderer_label,
            "rendererReady": self.renderer == "flexavatar" or self._panic_asset is not None,
            "hasSplat": self.renderer == "flexavatar",
            "panicInstalled": self._panic_backend.installed,
            "animeExpressionHandler": "tha4" if self._panic_backend.expression_handler_installed else "fallback",
            "animeSuperResolution": self._panic_backend.super_resolution_name,
            "animeSrCache": self._panic_backend.sr_stats(),
            "mode": self.mode,
            "playing": self.playing,
            "lockHead": self.lock_head,
            "points": points,
            "renderFps": self.render_fps,
            "animationFps": self.animation_fps,
            "webcamReady": self.webcam_ready,
            "lastError": self.last_error,
            "lastGeneratedAvatar": self.last_generated_avatar,
        }


class UploadResponse(BaseModel):
    avatarName: str
    path: str
    url: str


app = FastAPI(title="FlexAvatar Web Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session: Optional[WebAvatarSession] = None
exports_dir = Path(REPO_ROOT) / "data" / "web_exports"
exports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/exports", StaticFiles(directory=exports_dir), name="exports")
app.mount("/inputs", StaticFiles(directory=Path(FLEXAVATAR_INPUTS_PATH) / "itw"), name="inputs")


def get_session():
    if session is None:
        raise HTTPException(status_code=503, detail="FlexAvatar runtime is still starting")
    return session


@app.on_event("startup")
def startup():
    global session
    session = WebAvatarSession()


@app.get("/api/state")
def api_state():
    return get_session().state()


@app.get("/api/avatars")
def api_avatars():
    avatar_dir = Path(FLEXAVATAR_AVATAR_CODE_PATH) / "itw"
    avatars = []
    if avatar_dir.exists():
        for path in sorted(avatar_dir.glob("avatar_code_*.npy")):
            name = path.stem.removeprefix("avatar_code_")
            avatars.append({"name": name, "codePath": str(path)})
    return {"avatars": avatars}


@app.post("/api/upload", response_model=UploadResponse)
async def api_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "avatar.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".mp4"}:
        raise HTTPException(status_code=400, detail="Upload a .jpg, .png, or .mp4 file")
    avatar_name = WebAvatarSession.avatar_name_from_path(file.filename or "avatar")
    dest_suffix = ".jpg" if suffix == ".jpeg" else suffix
    dest = Path(FLEXAVATAR_INPUTS_PATH) / "itw" / f"{avatar_name}{dest_suffix}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())
    if session is not None:
        with session.lock:
            session.last_error = None
            session.last_generated_avatar = None
            session.status = f"Selected {avatar_name}. Ready to generate."
    return UploadResponse(avatarName=avatar_name, path=str(dest), url=f"/inputs/{dest.name}")


@app.post("/api/generate")
def api_generate(payload: dict, background_tasks: BackgroundTasks):
    current_session = get_session()
    if not current_session.prepare_generation():
        raise HTTPException(status_code=409, detail="Generation is already running")
    input_path = payload.get("path")
    if not input_path:
        current_session.busy = False
        raise HTTPException(status_code=400, detail="Missing input path")
    background_tasks.add_task(current_session.generate_avatar, input_path)
    return {"ok": True, "status": "Generation started"}


@app.post("/api/load/{avatar_name}")
def api_load_avatar(avatar_name: str):
    try:
        get_session()._load_avatar_for_name(avatar_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "state": get_session().state()}


@app.post("/api/controls")
def api_controls(payload: ControlPayload):
    get_session().apply_controls(payload)
    return {"ok": True, "state": get_session().state()}


@app.get("/api/frame.jpg")
def api_frame(width: int = 1280, height: int = 720):
    return Response(get_session().render_jpeg(width, height), media_type="image/jpeg")


@app.get("/api/frame.webp")
def api_frame_webp(width: int = 1280, height: int = 720):
    quality = int(os.environ.get("FLEXAVATAR_ANIME_WEBP_QUALITY", "96"))
    return Response(
        get_session().render_jpeg(width, height, quality=quality, output_format="WEBP"),
        media_type="image/webp",
    )


@app.get("/api/frame.png")
def api_frame_png(width: int = 1280, height: int = 720):
    return Response(get_session().render_jpeg(width, height, output_format="PNG"), media_type="image/png")


@app.get("/api/stream.mjpg")
async def api_stream(width: int = 1280, height: int = 720):
    async def frames():
        while True:
            frame_start = time.time()
            jpeg = get_session().render_jpeg(width, height, quality=88)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(max(0.0, (1 / 24) - (time.time() - frame_start)))

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/export-ply")
def api_export_ply():
    try:
        path = get_session().export_ply()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "url": f"/exports/{path.name}", "path": str(path)}


@app.post("/api/drive-frame")
async def api_drive_frame(file: UploadFile = File(...)):
    data = await file.read()
    try:
        get_session().drive_from_image(data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, "state": get_session().state()}


dist_dir = Path(REPO_ROOT) / "web" / "dist"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="web")


def main():
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
