import platform
from typing import List, Callable, Optional, Union, Tuple

import torch
import torchvision
from diffusers import AutoencoderKL
from dreifus.camera import PoseType
from dreifus.graphics import Dimensions
from dreifus.matrix import Pose, Intrinsics
from dreifus.vector import Vec3
from einops import rearrange
from elias.util.batch import batchify_sliced

from flexavatar.config.dataset_config import MVDatasetConfig, GaussianHeadLRMBatch, SampleMetadata
from flexavatar.model.dinov2 import DinoV2
from flexavatar.util.lru_cache import DeviceLRUCache


class GaussianHeadLRMPreprocessor:

    def __init__(self,
                 dataset_config: MVDatasetConfig,
                 use_caching: bool = False,
                 cache_dtype: torch.dtype = torch.float32,
                 cache_size: int = 50_000,
                 compile: bool = False,
                 use_bfloat16: bool = False,
                 finetuned_slrm_encoder: Optional[str] = None):
        self._dataset_config = dataset_config
        self._use_caching = use_caching
        self._cache_dtype = cache_dtype
        self._use_bfloat16 = use_bfloat16
        device = torch.device('cuda')

        if dataset_config.use_dino and not dataset_config.load_precomputed_dino:
            if dataset_config.use_dinov3:
                self._dino = DinoV3(dataset_config.dino_name, image_size=dataset_config.dino_resolution, extract_layers=dataset_config.extract_dino_layers)
            else:
                self._dino = DinoV2(dataset_config.dino_name)

            for p in self._dino.parameters():
                p.requires_grad = False

            if compile and platform.system() == 'Linux':
                self._dino.compile()

            if use_caching:
                self._dino_cache = DeviceLRUCache(device, torch.device('cpu'), cache_dtype=cache_dtype, max_size=cache_size)

        if dataset_config.use_vae:
            self._vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{dataset_config.vae_type}").to(device)

            if use_caching:
                self._vae_cache = DeviceLRUCache(device, torch.device('cpu'), cache_dtype=cache_dtype, max_size=cache_size)
                # self._cached_vae_images = defaultdict(dict)
                # self._cached_target_vae_images = defaultdict(dict)

            self._cached_render_bg_color = None

        if dataset_config.slrm_encoder is not None:
            if dataset_config.finetune_slrm_decoder and finetuned_slrm_encoder is not None:
                dit_model_manager = DiTModelFolder().open_run(finetuned_slrm_encoder)
                slrm_encoder_model = dit_model_manager.load_slrm_checkpoint(-1, dataset_config.slrm_encoder)
                self._slrm_encoder_model = slrm_encoder_model.to(device).eval()
            else:
                lifting_model_manager = StaticHeadLRMModelFolder().open_run(dataset_config.slrm_encoder)
                self._slrm_encoder_model = lifting_model_manager.load_checkpoint(-1).to(device).eval()

            if use_caching:
                self._slrm_encoder_cache = DeviceLRUCache(device, torch.device('cpu'), cache_dtype=cache_dtype, max_size=cache_size)

            if not dataset_config.finetune_slrm_decoder:
                for p in self._slrm_encoder_model.parameters():
                    p.requires_grad = False

    def process(self, batch: GaussianHeadLRMBatch, disable_cache: bool = False, enable_grads: bool = False):
        use_caching = self._use_caching and not disable_cache
        device = batch.input_images.device
        if self._dataset_config.use_dino and not self._dataset_config.load_precomputed_dino:
            batched_feature_images = []

            if use_caching:
                cached_idxs = []
                non_cached_idxs = []
                non_cached_images = []
                non_cached_sample_metadatas = []

                for b, input_sample_metadata in enumerate(batch.input_sample_metadatas):
                    feature_input_images = []
                    for v in range(len(input_sample_metadata)):
                        if input_sample_metadata[v] in self._dino_cache:
                            cached_idxs.append((b, v))
                            feature_image = self._dino_cache.get(input_sample_metadata[v])
                            feature_input_images.append(feature_image)
                        else:
                            non_cached_idxs.append((b, v))
                            non_cached_images.append(batch.input_images[b][v])
                            non_cached_sample_metadatas.append(input_sample_metadata[v])
                            feature_input_images.append(None)

                    batched_feature_images.append(feature_input_images)

                if non_cached_images:
                    non_cached_images = torch.stack(non_cached_images)
                    non_cached_features = self._get_dino_features(non_cached_images)

                    for (b, v), non_cached_sample_metadata, non_cached_feature in zip(non_cached_idxs, non_cached_sample_metadatas, non_cached_features):
                        self._dino_cache.put(non_cached_sample_metadata, non_cached_feature)
                        batched_feature_images[b][v] = non_cached_feature

                batched_feature_images = torch.stack([torch.stack(feature_images) for feature_images in batched_feature_images])

            else:
                for input_images in batch.input_images:
                    feature_images = self._get_dino_features(input_images, enable_grads=enable_grads)
                    batched_feature_images.append(feature_images)

                batched_feature_images = torch.stack(batched_feature_images)
            batch.features = batched_feature_images

        if self._dataset_config.use_vae:
            if self._cached_render_bg_color is None:
                with torch.no_grad():
                    bg_img = torch.ones_like(batch.input_images[0, 0]) * torch.tensor(batch.render_bg_color[0], device=batch.input_images.device)[:, None, None]
                    vae_bg = self._vae.encode(bg_img[None]).latent_dist.sample().mul_(0.18215)
                    vae_render_bg_color = vae_bg[0].mean(axis=2).mean(axis=1).cpu()
                    vae_render_bg_color = vae_render_bg_color * 255  # Counter-act division by 255 that happens inside model

                    if use_caching:
                        self._cached_render_bg_color = vae_render_bg_color.tolist()
            else:
                vae_render_bg_color = self._cached_render_bg_color

            batch.render_bg_color = [vae_render_bg_color for _ in range(len(batch.input_images))]

            batched_vae_images = []
            if use_caching:
                batched_vae_images = self._perform_cached_lookup(self._vae_cache, self._get_vae_image, batch.input_images, batch.input_sample_metadatas)
            else:
                with torch.no_grad():
                    for input_image in batch.input_images:
                        vae_image = self._vae.encode(input_image).latent_dist.sample().mul_(0.18215)
                        batched_vae_images.append(vae_image)

                batched_vae_images = torch.stack(batched_vae_images, dim=0)
            batch.input_images = batched_vae_images

            if batch.target_images is not None:
                batched_vae_target_images = self._perform_cached_lookup(self._vae_cache, self._get_vae_image, batch.target_images,
                                                                        batch.target_sample_metadatas)
                batch.target_images = batched_vae_target_images

            if batch.render_resolution is not None:
                batch.render_resolution = [Dimensions(dimension.x // 8, dimension.y // 8) for dimension in batch.render_resolution]
            batch.render_intrinsics = [[intr.rescale(1 / 8, inplace=False) for intr in intrinsics] for intrinsics in batch.render_intrinsics]

        if self._dataset_config.slrm_encoder is not None:

            if False and use_caching:
                # TODO: This does not yet work because _get_slrm_tokens takes two arguments
                batched_vae_images = self._perform_cached_lookup(self._slrm_encoder_cache, self._get_slrm_tokens, batch.input_images, batch.input_sample_metadatas)
            else:
                if not self._dataset_config.finetune_slrm_encoder:
                    no_grad_block = torch.no_grad()
                    no_grad_block.__enter__()

                if self._dataset_config.use_dino:
                    features = self._get_dino_features(batch.input_images.flatten(0, 1)).unflatten(0, (batch.input_images.shape[:2]))
                else:
                    features = None
                batched_slrm_encoder_images = self._slrm_encoder_model.create_gaussian_models(
                    images=batch.input_images,
                    input_cam2worlds=batch.input_cam2worlds,
                    input_intrinsics=batch.input_intrinsics,
                    features=features,
                    only_internal_representations=True)
                res = self._slrm_encoder_model._config.head_transformer.res_head_tokens
                d_hidden = self._slrm_encoder_model._config.head_transformer.transformer.d_hidden
                batched_slrm_encoder_images = batched_slrm_encoder_images.internal_representations.reshape(res, res, -1, d_hidden).permute(2, 3, 0, 1)[:, None]
                # batched_slrm_encoder_images = self._get_slrm_tokens(batch.input_images, batch.input_sample_metadatas)
                if not self._dataset_config.finetune_slrm_encoder:
                    no_grad_block.__exit__(None, None, None)

            batch.input_images = batched_slrm_encoder_images


        return batch

    def postprocess(self, rendered_images: torch.Tensor,
                    render_cam2world_poses: Optional[List[List[Pose]]] = None,
                    render_intrinsics: Optional[List[List[Intrinsics]]] = None,
                    render_resolutions: Optional[List[Union[Dimensions, int]]] = None,
                    render_bg_colors: Optional[List[Tuple[int, int, int]]] = None,
                    expression_codes: Optional[torch.Tensor] = None,
                    ):

        if self._dataset_config.use_vae:
            rendered_images_decoded = []
            with torch.no_grad():
                for rendered_image_batch in batchify_sliced(rendered_images, 1):
                    rendered_image_batch = self._vae.decode(rendered_image_batch.float() / 0.18215).sample
                    rendered_images_decoded.append(rendered_image_batch)
            rendered_images = torch.cat(rendered_images_decoded, dim=0)

        if self._dataset_config.slrm_encoder is not None:
            # self._slrm_encoder_model.forward(GaussianHeadLRMBatch(torch.empty((rendered_images.shape[0], 1, rendered_images.shape[1], rendered_images.shape[2], rendered_images.shape[3])), None, None, None, None, [[Pose()]], [[Intrinsics()]], [Dimensions(512, 512)], [(255, 255, 255)], None, expression_codes=torch.zeros((rendered_images.shape[0], 1, 126), device=rendered_images.device)), cached_internal_representations=rendered_images.permute(2, 3, 0, 1).flatten(0, 1))
            B = rendered_images.shape[0]

            is_multi_view = len(rendered_images.shape) == 5

            if render_cam2world_poses is None:
                frontal_cam2world = Pose(pose_type=PoseType.CAM_2_WORLD)
                frontal_cam2world.move(z=1)
                frontal_cam2world.look_at(Vec3(), up=Vec3(0, 1, 0))
                frontal_intrinsics = Intrinsics(1500, 1500, 256, 256)
                # frontal_intrinsics.rescale(1 / 512)
                render_cam2world_poses = [[frontal_cam2world] for _ in range(B)]
                render_intrinsics = [[frontal_intrinsics] for _ in range(B)]

            if expression_codes is None:
                expression_codes = torch.zeros((rendered_images.shape[0], 1, self._dataset_config.expression_code_config.get_dim()), device=rendered_images.device)
            if render_resolutions is None:
                render_resolutions = [Dimensions(512, 512) for _ in range(B)]
            if render_bg_colors is None:
                render_bg_colors = [(255, 255, 255) for _ in range(B)]
            batch = GaussianHeadLRMBatch(torch.empty((B, 1, rendered_images.shape[-3], rendered_images.shape[-2], rendered_images.shape[-1])),
                                     None, None, None, None, render_cam2world_poses, render_intrinsics, render_resolutions, render_bg_colors, None,
                                     expression_codes=expression_codes)
            if is_multi_view:
                cached_internal_representations = rearrange(rendered_images, 'b v c h w -> (v h w) b c')
            else:
                cached_internal_representations = rearrange(rendered_images, 'b c h w -> (h w) b c')
            output = self._slrm_encoder_model.forward(
                batch,
                cached_internal_representations=cached_internal_representations)

            if is_multi_view:
                rendered_images = output.rendering_output.rendered_images
            else:
                rendered_images = output.rendering_output.rendered_images[:, 0]

        return rendered_images

    def _get_dino_features(self, input_image: torch.Tensor, enable_grads: bool = False) -> torch.Tensor:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self._use_bfloat16):
            input_image = torchvision.transforms.Resize((self._dataset_config.dino_resolution, self._dataset_config.dino_resolution))(input_image)
            feature_image = self._dino(input_image, enable_grads=enable_grads)
        return feature_image

    def _get_vae_image(self, input_image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vae_image = self._vae.encode(input_image).latent_dist.sample().mul_(0.18215)
        return vae_image

    def _get_slrm_tokens(self, input_image: torch.Tensor, input_sample_metadatas) -> torch.Tensor:
        features = self._perform_cached_lookup(self._dino_cache, self._get_dino_features, input_image, input_sample_metadatas)
        slrm_tokens = self._slrm_encoder_model.create_gaussian_models(
            images=input_image,
            features=features,
            only_internal_representations=True)
        return slrm_tokens

    def _perform_cached_lookup(self,
                               cache: DeviceLRUCache,
                               transform_fn: Callable[[torch.Tensor], torch.Tensor],
                               images: torch.Tensor,
                               sample_metadatas: List[List[SampleMetadata]]):
        cached_idxs = []
        non_cached_idxs = []
        non_cached_images = []
        non_cached_sample_metadatas = []

        batched_transformed_images = []

        for b, sample_metadata in enumerate(sample_metadatas):
            transformed_images = []
            for v in range(len(sample_metadata)):
                if sample_metadata[v] in cache:
                    cached_idxs.append((b, v))
                    feature_image = cache.get(sample_metadata[v])
                    transformed_images.append(feature_image)
                else:
                    non_cached_idxs.append((b, v))
                    non_cached_images.append(images[b][v])
                    non_cached_sample_metadatas.append(sample_metadata[v])
                    transformed_images.append(None)

            batched_transformed_images.append(transformed_images)

        if non_cached_images:
            non_cached_images = torch.stack(non_cached_images)
            non_cached_transformed_images = transform_fn(non_cached_images)

            for (b, v), non_cached_sample_metadata, non_cached_transformed_image in zip(non_cached_idxs, non_cached_sample_metadatas,
                                                                                        non_cached_transformed_images):
                cache.put(non_cached_sample_metadata, non_cached_transformed_image)
                batched_transformed_images[b][v] = non_cached_transformed_image

        batched_transformed_images = torch.stack(
            [torch.stack(transformed_images) if transformed_images else torch.empty((0,)) for transformed_images in batched_transformed_images])

        return batched_transformed_images
