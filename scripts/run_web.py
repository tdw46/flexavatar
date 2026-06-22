import asyncio
import io
import os
import ssl
import time
import traceback
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
from PIL import Image
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


class WebAvatarSession:
    def __init__(self):
        self.lock = RLock()
        self.status = "Booting FlexAvatar web runtime..."
        self.current_avatar = "marble_sculpture"
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

        self.camera = WebCamera()
        self.gaussians = GaussianModel(3)
        self._window_closed = False
        self._sheap_module = None

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
            self.current_avatar = avatar_name
            self._batch = batch
            self._avatar_code = output.internal_representations
            self.gaussians = output.gaussian_models[0][0]

    def generate_avatar(self, input_path: str):
        if not self.busy and not self.prepare_generation():
            return
        try:
            avatar_name, copied_path = self._copy_avatar_input(input_path)
            self._set_status(f"Tracking {avatar_name} with Pixel3DMM...")
            run_pixel3dmm(copied_path)

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
            for index, value in enumerate(payload.expression[:100]):
                self._manual_expression_code[index] = float(value)
            self._apply_rotation(payload.jaw[:3], 120)
            self._apply_rotation(payload.head[:3], 126)
            camera = payload.camera or {}
            self.camera.yaw = float(camera.get("yaw", self.camera.yaw))
            self.camera.pitch = float(camera.get("pitch", self.camera.pitch))
            self.camera.roll = float(camera.get("roll", self.camera.roll))
            self.camera.radius = float(camera.get("radius", self.camera.radius))
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

            with self.lock:
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

    def render_jpeg(self, width: int = 1280, height: int = 720, quality: int = 90):
        start = time.time()
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

    def export_ply(self):
        with self.lock:
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

    def state(self):
        points = int(self.gaussians._xyz.shape[0]) if self.gaussians._xyz is not None else 0
        return {
            "status": self.status,
            "busy": self.busy,
            "currentAvatar": self.current_avatar,
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


@app.get("/api/stream.mjpg")
async def api_stream(width: int = 1280, height: int = 720):
    async def frames():
        while True:
            jpeg = get_session().render_jpeg(width, height, quality=82)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(1 / 20)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/export-ply")
def api_export_ply():
    path = get_session().export_ply()
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
