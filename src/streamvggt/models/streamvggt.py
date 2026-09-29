import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub
import numpy as np

from streamvggt.models.aggregator import Aggregator
from streamvggt.heads.camera_head import CameraHead
from streamvggt.heads.dpt_head import DPTHead
from streamvggt.heads.track_head import TrackHead
from streamvggt.utils.dom_state import DOMStateManager, DOMState, DOMStateConfig
from streamvggt.utils.tta_dom import CameraHeadTTA, TTAConfig
from streamvggt.utils.geo_ttt import GeoTTT, GeoTTTConfig
from streamvggt.geo_v3 import GeoV3Config, GeoV3Runner, default_segment_infer_fn
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple, List, Any, Callable
from dataclasses import dataclass

@dataclass
class StreamVGGTOutput(ModelOutput):
    ress: Optional[List[dict]] = None
    views: Optional[torch.Tensor] = None

class StreamVGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
    


    def forward(
        self,
        views,
        query_points: torch.Tensor = None,
        history_info: Optional[dict] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ):
        images = torch.stack(
            [view["img"] for view in views], dim=0
        ).permute(1, 0, 2, 3, 4)    # B S C H W

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if history_info is None:
            history_info = {"token": None}

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.track_head is not None and query_points is not None:
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                predictions["track"] = track_list[-1]  # track of the last iteration
                predictions["vis"] = vis
                predictions["conf"] = conf
            predictions["images"] = images

            B, S = images.shape[:2]
            ress = []
            for s in range(S):
                res = {
                    'pts3d_in_other_view': predictions['world_points'][:, s],  # [B, H, W, 3]
                    'conf': predictions['world_points_conf'][:, s],  # [B, H, W]

                    'depth': predictions['depth'][:, s],  # [B, H, W, 1]
                    'depth_conf': predictions['depth_conf'][:, s],  # [B, H, W]
                    'camera_pose': predictions['pose_enc'][:, s, :],  # [B, 9]

                    **({'valid_mask': views[s]["valid_mask"]}
                    if 'valid_mask' in views[s] else {}),  # [B, H, W]

                    **({'track': predictions['track'][:, s],  # [B, N, 2]
                        'vis': predictions['vis'][:, s],  # [B, N]
                        'track_conf': predictions['conf'][:, s]}
                    if 'track' in predictions else {})
                }
                ress.append(res)
            return StreamVGGTOutput(ress=ress, views=views)  # [S] [B, C, H, W]
        
    def inference(self, frames, query_points: torch.Tensor = None, past_key_values=None):        
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth
        
        all_ress = []
        processed_frames = []

        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0) 
            aggregator_output = self.aggregator(
                images, 
                past_key_values=past_key_values,
                use_cache=True, 
                past_frame_idx=i
            )
            
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output
            
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(aggregated_tokens, past_key_values_camera=past_key_values_camera, use_cache=True)
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    depth = depth[:, 0] 
                    depth_conf = depth_conf[:, 0]
                
                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx
                    )
                    pts3d = pts3d[:, 0] 
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens, images=images, patch_start_idx=patch_start_idx, query_points=query_points
                )
                    track = track_list[-1][:, 0]  
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            all_ress.append({
                'pts3d_in_other_view': pts3d,
                'conf': pts3d_conf,
                'depth': depth,
                'depth_conf': depth_conf,
                'camera_pose': camera_pose,
                **({'valid_mask': frame["valid_mask"]}
                    if 'valid_mask' in frame else {}),  

                **({'track': track, 
                    'vis': vis,  
                    'track_conf': track_conf}
                if query_points is not None else {})
            })
            processed_frames.append(frame)
        
        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        return output

    def inference_with_dom(
        self,
        frames,
        dom_image: torch.Tensor,
        project_fn: Callable,
        anchor_world_xyz: torch.Tensor,
        get_world_xyz_fn: Callable,
        query_points: torch.Tensor = None,
        dom_config: DOMStateConfig = None,
        dom_state_carry: DOMState = None,
        heading_fn: Callable = None,
        enable_dom_as_frame: bool = True,
    ):
        """
        Streaming inference with DOM reference as an extra input frame.

        Strategy 1 — DOM-as-frame:
          The DOM crop (at camera footprint scale) is fed to the aggregator as
          frame 0.  All subsequent camera frames attend to the DOM through the
          aggregator's global cross-frame attention, giving the model absolute
          spatial reference without modifying any learned parameters.

        Strategy 2 — NCC snap-back:
          Every `snap_interval` frames the predicted pose is corrected by
          image-space NCC matching between the camera image and a 2× footprint
          DOM crop.  The correction is applied directly to the pose translation
          (converted via cam_gsd, no numerical Jacobian).

        Args:
            frames: List of frame dicts with "img" key [1, 3, H, W].
            dom_image: Full DOM [B, 3, H_dom, W_dom] (may be on CPU).
            project_fn: world_xyz [B, 3] → DOM UV [B, 2].
            anchor_world_xyz: First frame's world position [B, 3].
            get_world_xyz_fn: pose_params [B, 9] → world_xyz [B, 3].
            query_points: Optional tracking query.
            dom_config: DOMStateConfig override.
            dom_state_carry: DOMState from previous window.
            heading_fn: frame_idx → heading_rad.

        Returns:
            StreamVGGTOutput with corrected poses and diagnostics.
        """
        cfg = dom_config or DOMStateConfig()
        dom_mgr = DOMStateManager(cfg)
        dom_state = dom_state_carry if dom_state_carry is not None else DOMState(cfg)

        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth

        all_ress = []
        processed_frames = []
        dom_diagnostics = []

        # ---- Strategy 1: Feed DOM crop as frame 0 (optional) ----
        frame_idx_offset = 0
        if enable_dom_as_frame:
            cam_hw = (frames[0]["img"].shape[-2], frames[0]["img"].shape[-1])
            dom_crop = dom_mgr._crop_dom_at_camera_scale(
                dom_image, anchor_world_xyz, project_fn, cam_hw,
            )  # [B, 3, H_cam, W_cam]
            dom_crop = dom_crop.to(frames[0]["img"].device)

            dom_images = dom_crop.unsqueeze(0)  # [B=1, S=1, 3, H, W]
            dom_agg_output = self.aggregator(
                dom_images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=0,
            )
            if isinstance(dom_agg_output, tuple) and len(dom_agg_output) == 3:
                _, patch_start_idx, past_key_values = dom_agg_output
            else:
                _, patch_start_idx = dom_agg_output

            frame_idx_offset = 1
            dom_diagnostics.append({"frame_idx": -1, "dom_action": "dom_as_frame0"})
        else:
            dom_diagnostics.append({"frame_idx": -1, "dom_action": "dom_as_frame_skipped"})

        # ---- Process camera frames ----
        for i, frame in enumerate(frames):
            images = frame["img"].unsqueeze(0)
            aggregator_output = self.aggregator(
                images,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=i + frame_idx_offset,
            )

            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens, patch_start_idx = aggregator_output

            frame_diag = {"frame_idx": i}

            # ---- Run heads ----
            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc, past_key_values_camera = self.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=past_key_values_camera,
                        use_cache=True,
                    )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]  # [B, 9]

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                    depth = depth[:, 0]
                    depth_conf = depth_conf[:, 0]

                if self.point_head is not None:
                    pts3d, pts3d_conf = self.point_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                    )
                    pts3d = pts3d[:, 0]
                    pts3d_conf = pts3d_conf[:, 0]

                if self.track_head is not None and query_points is not None:
                    track_list, vis, conf = self.track_head(
                        aggregated_tokens,
                        images=images,
                        patch_start_idx=patch_start_idx,
                        query_points=query_points,
                    )
                    track = track_list[-1][:, 0]
                    query_points = track
                    vis = vis[:, 0]
                    track_conf = conf[:, 0]

            # ---- Strategy 2: NCC snap-back every snap_interval frames ----
            delta_xy = None
            dom_confidence = torch.zeros(camera_pose.shape[0], device=camera_pose.device)

            if i > 0 and cfg.pose_xy_blend_strength > 0:
                current_world_xyz = get_world_xyz_fn(camera_pose)

                heading_rad = 0.0
                if heading_fn is not None:
                    heading_rad = heading_fn(i)

                delta_xy, dom_confidence = dom_mgr.compute_ncc_snapback(
                    dom_image, current_world_xyz,
                    images[:, 0],  # [B, 3, H, W]
                    heading_rad,
                    project_fn,
                )

                if delta_xy is not None:
                    camera_pose = dom_mgr.blend_pose_xy(
                        camera_pose, delta_xy, dom_confidence, dom_state,
                    )

                frame_diag["dom_xy_correction"] = (
                    delta_xy.tolist() if delta_xy is not None else None
                )
                frame_diag["dom_search_confidence"] = (
                    dom_confidence.item() if dom_confidence.numel() == 1
                    else dom_confidence.tolist()
                )

            # ---- Accumulate results ----
            all_ress.append({
                "pts3d_in_other_view": pts3d,
                "conf": pts3d_conf,
                "depth": depth,
                "depth_conf": depth_conf,
                "camera_pose": camera_pose,
                **({
                    "valid_mask": frame["valid_mask"]
                } if "valid_mask" in frame else {}),
                **({
                    "track": track,
                    "vis": vis,
                    "track_conf": track_conf,
                } if query_points is not None else {}),
            })
            processed_frames.append(frame)
            dom_diagnostics.append(frame_diag)
            dom_state.frame_idx += 1

        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        output.dom_diagnostics = dom_diagnostics
        output.dom_state = dom_state
        return output

    def inference_with_tta(
        self,
        frames,
        dom_image: torch.Tensor,
        project_fn: Callable,
        anchor_world_xyz: torch.Tensor,
        get_world_xyz_fn: Callable,
        query_points: torch.Tensor = None,
        tta_config: TTAConfig = None,
        tta_manager: CameraHeadTTA = None,
        heading_fn: Callable = None,
        dtype=torch.bfloat16,
    ):
        """
        Streaming inference with test-time adaptation of camera_head.

        At the start of this window, performs K gradient steps on camera_head's
        poseLN_modulation using NCC loss between camera images and DOM crops.
        Then runs normal streaming inference with the adapted parameters.

        Args:
            frames: List of frame dicts with "img" key [1, 3, H, W].
            dom_image: Full DOM [B, 3, H_dom, W_dom] (may be on CPU).
            project_fn: world_xyz [B, 3] -> DOM UV [B, 2].
            anchor_world_xyz: First frame's world position [B, 3].
            get_world_xyz_fn: pose_params [B, 9] -> world_xyz [B, 3].
            query_points: Optional tracking query.
            tta_config: TTAConfig override.
            tta_manager: CameraHeadTTA instance (carries adapted state across windows).
            heading_fn: frame_idx -> heading_rad.
            dtype: compute dtype for inference.

        Returns:
            StreamVGGTOutput with poses from adapted camera_head, plus tta_diagnostics.
        """
        cfg = tta_config or TTAConfig()

        # Create TTA manager if not provided (first window)
        if tta_manager is None:
            tta_manager = CameraHeadTTA(cfg, self.camera_head)

        # --- Step 1: TTA adaptation ---
        tta_diag = tta_manager.adapt_window(
            model=self,
            frames=frames,
            dom_image=dom_image,
            project_fn=project_fn,
            anchor_world_xyz=anchor_world_xyz,
            get_world_xyz_fn=get_world_xyz_fn,
            heading_fn=heading_fn,
            dtype=dtype,
        )

        # Free TTA computation graph memory before inference
        torch.cuda.empty_cache()

        # --- Step 2: Normal streaming inference with adapted camera_head ---
        # Only run camera_head (skip depth/point/track heads to save GPU memory)
        past_key_values = [None] * self.aggregator.depth
        past_key_values_camera = [None] * self.camera_head.trunk_depth

        all_ress = []
        processed_frames = []

        with torch.no_grad():
            for i, frame in enumerate(frames):
                images = frame["img"].unsqueeze(0)
                with torch.cuda.amp.autocast(dtype=dtype):
                    aggregator_output = self.aggregator(
                        images,
                        past_key_values=past_key_values,
                        use_cache=True,
                        past_frame_idx=i,
                    )

                if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                    aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
                else:
                    aggregated_tokens, patch_start_idx = aggregator_output

                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc, past_key_values_camera = self.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=past_key_values_camera,
                        use_cache=True,
                    )
                    pose_enc = pose_enc[-1]
                    camera_pose = pose_enc[:, 0, :]

                all_ress.append({
                    "camera_pose": camera_pose,
                })
                processed_frames.append(frame)

        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        output.tta_diagnostics = tta_diag
        output.tta_manager = tta_manager
        return output

    def inference_with_geo_ttt(
        self,
        frames,
        dom_image: torch.Tensor,
        project_fn: Callable,
        anchor_world_xyz: torch.Tensor,
        get_world_xyz_fn: Callable,
        query_points: torch.Tensor = None,
        geo_ttt_config: GeoTTTConfig = None,
        geo_ttt_manager: GeoTTT = None,
        heading_fn: Callable = None,
        dtype=torch.bfloat16,
    ):
        """
        Streaming inference with Geo-supervised KV-Cache TTT.

        想法5: Treats camera head's KV-cache as TTT3R-style fast weights,
        updated via DOM reprojection loss gradient descent:
            KV_cam <- KV_cam - eta * grad_KV L_dom

        After TTT adaptation, poses are re-computed using the modified
        KV-cache and adapted poseLN — no redundant aggregator re-run.

        Falls back to standard inference if TTT is skipped (too few matches).
        """
        cfg = geo_ttt_config or GeoTTTConfig()

        if geo_ttt_manager is None:
            geo_ttt_manager = GeoTTT(cfg, self.camera_head)

        # --- Step 1: Geo-TTT adaptation (also caches agg_outputs internally) ---
        geo_ttt_diag = geo_ttt_manager.adapt_window(
            model=self,
            frames=frames,
            dom_image=dom_image,
            project_fn=project_fn,
            anchor_world_xyz=anchor_world_xyz,
            get_world_xyz_fn=get_world_xyz_fn,
            heading_fn=heading_fn,
            dtype=dtype,
        )

        torch.cuda.empty_cache()

        all_ress = []
        processed_frames = []

        if not geo_ttt_diag.get("skipped", False):
            # --- Step 2a: Use adapted KV-cache + poseLN for pose computation ---
            adapted_poses = geo_ttt_manager.compute_adapted_poses(
                self.camera_head,
            )
            for i, frame in enumerate(frames):
                all_ress.append({"camera_pose": adapted_poses[i]})
                processed_frames.append(frame)
        else:
            # --- Step 2b: Fallback to standard streaming inference ---
            past_key_values = [None] * self.aggregator.depth
            past_key_values_camera = [None] * self.camera_head.trunk_depth

            with torch.no_grad():
                for i, frame in enumerate(frames):
                    images = frame["img"].unsqueeze(0)
                    with torch.cuda.amp.autocast(dtype=dtype):
                        aggregator_output = self.aggregator(
                            images,
                            past_key_values=past_key_values,
                            use_cache=True,
                            past_frame_idx=i,
                        )

                    if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                        aggregated_tokens, patch_start_idx, past_key_values = aggregator_output
                    else:
                        aggregated_tokens, patch_start_idx = aggregator_output

                    with torch.cuda.amp.autocast(enabled=False):
                        pose_enc, past_key_values_camera = self.camera_head(
                            aggregated_tokens,
                            past_key_values_camera=past_key_values_camera,
                            use_cache=True,
                        )
                        pose_enc = pose_enc[-1]
                        camera_pose = pose_enc[:, 0, :]

                    all_ress.append({"camera_pose": camera_pose})
                    processed_frames.append(frame)

        output = StreamVGGTOutput(ress=all_ress, views=processed_frames)
        output.geo_ttt_diagnostics = geo_ttt_diag
        output.geo_ttt_manager = geo_ttt_manager
        return output

    def inference_with_geo_v3(
        self,
        frames,
        dom_image: torch.Tensor,
        project_fn: Callable,
        anchor_world_xyz: torch.Tensor,
        get_world_xyz_fn: Callable,
        query_points: torch.Tensor = None,
        image_paths=None,
        inv_project_fn=None,
        geo_elev=None,
        dom_transform=None,
        roma_model=None,
        save_vis_dir=None,
        geo_v3_config: GeoV3Config = None,
        geo_v3_runner: GeoV3Runner = None,
        geo_v3_metadata: dict = None,
        heading_fn: Callable = None,
        dtype=torch.bfloat16,
        segment_infer_fn: Callable = None,
        gt_first_frame_rotation=None,
    ):
        """GeoTTT-v3 entry point: segment hard reset + model-space TTT."""
        cfg = geo_v3_config or GeoV3Config()
        runner = geo_v3_runner or GeoV3Runner(cfg)

        if segment_infer_fn is None:
            segment_infer_fn = default_segment_infer_fn

        num_frames = len(frames)
        gt_enu = np.zeros((num_frames, 3), dtype=np.float32)
        global_traj, chain = runner.run(
            model=self,
            frames=frames,
            gt_enu=gt_enu,
            segment_infer_fn=segment_infer_fn,
            dom_image=dom_image,
            project_fn=project_fn,
            anchor_world_xyz=anchor_world_xyz,
            get_world_xyz_fn=get_world_xyz_fn,
            query_points=query_points,
            image_paths=image_paths,
            inv_project_fn=inv_project_fn,
            geo_elev=geo_elev,
            dom_transform=dom_transform,
            roma_model=roma_model,
            save_vis_dir=save_vis_dir,
            heading_fn=heading_fn,
            dtype=dtype,
            gt_first_frame_rotation=gt_first_frame_rotation,
            **(geo_v3_metadata or {}),
        )

        output = StreamVGGTOutput(ress=[], views=frames)
        output.geo_v3_chain = chain
        output.geo_v3_trajectory = global_traj
        output.geo_v3_config = cfg
        return output