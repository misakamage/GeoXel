from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GeoV3Config:
    """Configuration for GeoTTT-v3 segment-based adaptation.

    This config accepts both the native geo_v3 names and the geo_v2-style
    command-line semantics so existing eval commands can map over directly.
    """

    segment_length: int = 32
    overlap: int = 4
    warmup_frames: int = 2

    # GeoV2-compatible interface
    geo_ttt_v2_mode: str = "pose_ttt_token"
    geo_ttt_v2_steps: int = 3
    geo_ttt_v2_kv_lr: float = 1e-2
    geo_ttt_v2_reg_weight: float = 0.01
    geo_ttt_v2_loss: str = "mse"
    geo_consist_stride: int = 1
    geo_consist_max_corr: int = 500
    mse_weight: float = 1.0
    heading_weight: float = 0.0
    dense_reproj_weight: float = 0.5
    dense_reproj_z_weight: float = 0.1
    dense_reproj_huber_xy: float = 5.0
    dense_reproj_huber_z: float = 10.0
    # Plan C: pixel-level reprojection residual on road-only ground points.
    dense_reproj_pixel_weight: float = 0.0
    dense_reproj_pixel_huber: float = 5.0
    cam_token_weight: float = 10.0
    max_frames_for_loss: int = 8

    # Optimization aliases used by the current implementation
    kv_lr: float = 1e-2
    num_steps: int = 3
    reg_weight: float = 0.01
    loss_type: str = "mse"

    # Geometry / anchoring
    anchor_stride: int = 1

    # Reset / chaining policy
    hard_reset: bool = True
    use_segment_overlap: bool = True
    chain_only_outputs: bool = True

    # DOM crop
    dom_crop_padding_px: int = 500

    # Overlap frame continuity loss (TTT inner loop)
    overlap_weight: float = 1.0

    # When the head-overlap frames give a low-residual SRT into ENU, use that
    # SRT only as a bootstrap/crop prior for the start of the new segment.  It
    # is not a downstream primary transform, DOM refit gate, PGO gate, or Z gate.
    overlap_prior_bootstrap_enable: bool = True
    # Deprecated compatibility name; internally this now means bootstrap enable.
    overlap_prior_primary_enable: bool = True
    overlap_prior_primary_max_residual_m: float = 2.0
    overlap_prior_primary_min_frames: int = 3
    overlap_prior_restrict_strong_supervision: bool = False
    overlap_prior_consistency_max_xy_m: float = 25.0
    overlap_prior_dom_refit_min_new_frames: int = 3
    overlap_prior_dom_refit_gate_enable: bool = True
    # Deprecated/ignored: overlap prior must not lock DOM refit scale.
    overlap_prior_dom_refit_fixed_scale_fallback: bool = False
    overlap_prior_dom_refit_scale_ratio_min: float = 0.75
    overlap_prior_dom_refit_scale_ratio_max: float = 1.35
    overlap_prior_dom_refit_max_heading_deg: float = 20.0
    overlap_prior_dom_refit_max_overlap_drift_m: float = 5.0
    overlap_prior_cache_reliable_only: bool = False
    overlap_prior_pose_from_cache: bool = True
    overlap_prior_pose_min_frames: int = 1
    overlap_prior_gate_pgo: bool = False
    overlap_prior_pgo_max_omega_rad: float = 0.35
    overlap_prior_pgo_max_overlap_drift_m: float = 5.0
    overlap_prior_pgo_max_t_delta_m: float = 30.0
    overlap_prior_gate_z_correct: bool = False
    overlap_prior_z_correct_max_abs_m: float = 50.0
    overlap_prior_z_correct_max_std_m: float = 80.0

    # DEM-Z from matched road DEM points. The legacy enable flag is the master
    # switch; direct segment translation and global PGO edge export can be
    # split for ablations. Non-building samples are diagnostic-only and must
    # not be exported as DEM-Z PGO edges.
    dem_z_correct_enable: bool = True
    dem_z_direct_correct_enable: bool = True
    dem_z_pgo_edges_enable: bool = True
    dem_z_camera_correct_enable: bool = False
    dem_z_camera_pgo_edges_enable: bool = False
    dem_z_road_direct_correct_enable: bool = True
    dem_z_diagnostic_only: bool = False
    dem_z_correct_min_frames: int = 5
    dem_z_correct_min_points: int = 50
    dem_z_correct_max_abs_m: float = 20.0
    dem_z_correct_max_std_m: float = 8.0
    dem_z_correct_allow_nonbuilding_direct: bool = False  # deprecated; ignored by road-only DEM-Z

    # Plan A: restrict per-frame DOM supervision (pose + dense reproj) to
    # overlap frames only (head + tail `overlap` frames of each segment).
    # Middle frames are left untouched inside the segment -- they are driven
    # by StreamVGGT forward + smoothness + overlap continuity only.
    dom_supervision_overlap_only: bool = False

    # Diagnostics
    max_grad_norm: float = 0.0

    # In-segment PGO (Method-A: KV-only TTT then 1 segment-level scipy LS over (omega, t, delta);
    # log_s frozen). When enabled, replaces the fused-Adam path entirely.
    seg_pgo_enable: bool = False
    seg_pgo_w_dom_xy: float = 5.0
    seg_pgo_w_dom_z: float = 2.0
    seg_pgo_w_consist: float = 8.0
    seg_pgo_use_overlap_consistency: bool = False
    seg_pgo_w_delta_reg: float = 2.0
    seg_pgo_w_pose_rot: float = 50.0
    seg_pgo_w_pose_trans: float = 2.0
    seg_pgo_yaw_only_rotation: bool = False
    seg_pgo_freeze_delta: bool = True
    seg_pgo_use_point_edges: bool = False
    seg_pgo_w_point_xy: float = 0.5
    seg_pgo_w_point_z: float = 0.1
    seg_pgo_max_iter: int = 20
    seg_pgo_delta_clip: float = 5.0
    seg_pgo_verbose: bool = False
    seg_pgo_gate_on_overlap_bootstrap: bool = True
    seg_pgo_bootstrap_max_omega_rad: float = 0.25
    seg_pgo_bootstrap_max_t_delta_m: float = 15.0
    seg_pgo_bootstrap_max_overlap_motion_m: float = 5.0
    seg_pgo_bootstrap_max_prev_overlap_xy_m: float = 5.0

    # Direct DOM/DEM point correspondences sampled from matched query pixels.
    # Stored per segment and consumed by SegmentPGO / global PGO when their
    # corresponding point-edge weights are enabled.
    dom_point_edges_enable: bool = False
    dom_point_edges_road_only: bool = False
    dom_point_edges_max: int = 512
    # dom_points-only Z fallback: when road DEM-Z samples disappear, let global
    # PGO use non-building matched DEM-Z samples weakly instead of having no
    # local terrain-height evidence for late segments.
    dom_point_z_nonbuilding_fallback_enable: bool = False
    dom_point_bootstrap_enable: bool = False
    dom_point_prefer_point_bootstrap_transform: bool = True
    dom_point_bootstrap_require_overlap_consistency: bool = False
    dom_point_bootstrap_min_overlap_consistency: int = 1
    dom_point_bootstrap_keep_pose_orientation: bool = False
    dom_point_bootstrap_max_points: int = 4096
    dom_point_bootstrap_anchor_align: bool = True
    dom_point_bootstrap_prefer_overlap_frames: bool = True
    dom_point_bootstrap_camera_recenter_enable: bool = True
    dom_point_bootstrap_camera_recenter_segment0_enable: bool = False
    dom_point_bootstrap_camera_recenter_max_pitch_deg: float = 12.0
    dom_point_bootstrap_camera_recenter_min_frames: int = 3
    dom_point_bootstrap_camera_recenter_inlier_m: float = 15.0
    dom_point_bootstrap_camera_recenter_max_apply_xy_m: float = 80.0
    dom_point_bootstrap_camera_recenter_xy_weight: float = 1.0
    dom_point_bootstrap_agl_z_recenter_enable: bool = True
    dom_point_bootstrap_agl_z_weight: float = 1.0
    dom_point_bootstrap_agl_z_max_apply_m: float = 80.0
    dom_point_bootstrap_agl_z_inlier_m: float = 15.0
    # Deprecated compatibility knob; previous-segment overlap geometry is not a scale source.
    dom_point_bootstrap_inherit_overlap_scale: bool = False
    dom_point_srt_enable: bool = False
    dom_point_srt_rematch: bool = False
    dom_point_srt_ransac_iters: int = 256
    dom_point_srt_min_inliers: int = 32
    dom_point_srt_min_ratio: float = 0.05
    dom_point_srt_thresh_m: float = 15.0
    dom_point_srt_z_weight: float = 0.25
    dom_point_srt_scale_ratio_min: float = 0.5
    dom_point_srt_scale_ratio_max: float = 2.0
    dom_point_srt_max_rot_deg: float = 60.0

    # PointSRT-as-teacher TTT: robust DOM/DEM point correspondences are used
    # to build token-space supervision before decoding final poses/points.
    # Unlike dom_point_srt_enable, this path does not directly overwrite the
    # segment transform; the correction is absorbed through token adaptation.
    dom_point_ttt_enable: bool = False
    dom_point_ttt_use_srt_teacher: bool = True
    dom_point_ttt_raw_gate_enable: bool = True
    dom_point_ttt_raw_max_median_xy_m: float = 30.0
    dom_point_ttt_raw_max_p90_xy_m: float = 80.0
    dom_point_ttt_raw_min_inlier_ratio: float = 0.25
    dom_point_ttt_pose_teacher_source: str = "srt"
    dom_point_ttt_road_only: bool = False
    dom_point_ttt_max_points: int = 2048
    dom_point_ttt_min_inliers: int = 64
    dom_point_ttt_min_ratio: float = 0.25
    dom_point_ttt_max_median_xy_m: float = 3.0
    dom_point_ttt_w_point_xy: float = 0.25
    dom_point_ttt_w_point_z: float = 0.02
    dom_point_ttt_w_pose_xy: float = 2.0
    dom_point_ttt_w_pose_z: float = 0.0
    dom_point_ttt_point_huber_m: float = 5.0
    dom_point_ttt_pose_huber_m: float = 10.0
    dom_point_ttt_replace_dense: bool = True
    dom_point_ttt_pose_all_frames: bool = True
    dom_point_ttt_disable_direct_srt: bool = True
    # TTT-only camera-center pose teacher. It does not create PGO camera unary
    # edges. "fixedR_pnp" uses accepted fixed-rotation PnP XY and keeps current Z.
    dom_point_ttt_camera_teacher_source: str = "dom_points"
    # dom_points mode: camera target source used for segment/global PGO unary
    # edges. "dom_points" keeps the point-derived self-consistency target.
    dom_point_camera_teacher_enable: bool = False
    dom_point_camera_teacher_source: str = "dom_points"
    dom_point_camera_teacher_pgo_enable: bool = True
    dom_point_camera_teacher_min_points: int = 8
    dom_point_camera_teacher_max_points_per_frame: int = 256
    dom_point_camera_teacher_point_ray_ransac_iters: int = 128
    dom_point_camera_teacher_point_ray_inlier_thresh_m: float = 6.0
    dom_point_camera_teacher_point_ray_min_ratio: float = 0.25
    dom_point_camera_teacher_point_ray_max_median_m: float = 4.0
    dom_point_camera_teacher_point_ray_max_p90_m: float = 10.0
    dom_point_camera_teacher_point_ray_max_prior_xy_m: float = 50.0
    dom_point_camera_teacher_point_ray_max_prior_z_m: float = 120.0
    dom_point_camera_teacher_point_ray_use_z: bool = False
    dom_point_pose_crop_scale: float = 1.4

    # dom_points-only quality routing. Good DOM point SRT keeps the existing
    # point-ray/frame-center path; bad point SRT can fall back to accepted
    # fixed-rotation PnP camera centers, and bad point edges are not exported.
    dom_point_adaptive_pnp_enable: bool = True
    dom_point_quality_gate_enable: bool = True
    dom_point_quality_min_inlier_ratio: float = 0.65
    dom_point_quality_max_median_xy_m: float = 5.0
    dom_point_quality_min_road_frame_ratio: float = 0.0
    dom_point_pnp_fallback_min_frames: int = 3
    dom_point_pnp_fallback_min_inliers: int = 128
    dom_point_pnp_fallback_min_positive_ratio: float = 0.95
    dom_point_pnp_fallback_max_reproj_px: float = 30.0
    dom_point_pnp_fallback_max_ray_residual_m: float = 2.0
    dom_point_pnp_fallback_max_prior_xy_m: float = 40.0
    dom_point_pnp_fallback_gate_prior_xy: bool = True

    # dom_points-only camera trajectory constraint. The DOM frame center is a
    # ground point, not a camera center: the predicted optical-axis ray should
    # intersect the ground plane at frame_center_enu.xy.
    dom_point_ground_ray_enable: bool = False
    dom_point_ground_ray_ttt_enable: bool = False
    dom_point_ground_ray_seg_pgo_enable: bool = False
    dom_point_ground_ray_pgo_enable: bool = False
    dom_point_ground_ray_ttt_w_xy: float = 2.0
    dom_point_ground_ray_ttt_huber_m: float = 10.0
    dom_point_ground_ray_seg_pgo_w_xy: float = 2.0
    dom_point_ground_ray_pgo_w_xy: float = 2.0
    dom_point_ground_ray_min_abs_dir_z: float = 0.05

    # dom_points-only observation semantics. Failed pose crops are diagnostics,
    # not target observations; segment PGO may re-anchor from absolute DOM/ray
    # evidence when overlap continuity is already contradicted.
    dom_point_observation_validity_enable: bool = True
    dom_point_overlap_cache_reliable_only: bool = True
    dom_point_bootstrap_filter_head_overlap: bool = True
    dom_point_seg_pgo_absolute_reanchor_enable: bool = False
    dom_point_seg_pgo_reanchor_max_cost_ratio: float = 0.50
    dom_point_seg_pgo_reanchor_min_points: int = 32
    dom_point_seg_pgo_reanchor_min_ground_rays: int = 4

    # Prefer ground-footprint displacement parsed from DOM matches as canonical
    # camera XY correction.  This is more observable than directly trusting a
    # fixed-R PnP camera center under pitch / near-planar geometry.
    geo_target_mode: str = "auto"
    prefer_ground_signal_targets: bool = True
    ground_signal_max_delta_xy_m: float = 120.0
    ground_signal_min_coverage: float = 0.01
    segment_target_robust_fit: bool = True

    # Historical switch name kept for compatibility: it enables pose-derived
    # DOM targets, but raw fixed-R PnP camera centers are no longer used unless
    # allow_raw_pnp_camera_targets is explicitly enabled.
    prefer_pnp_camera_targets: bool = True
    allow_raw_pnp_camera_targets: bool = False

    # Deprecated compatibility knobs. The old pitch gate no longer decides
    # target semantics; prefer_pnp_camera_targets is the active switch.
    use_pnp_gate: bool = False
    pnp_pitch_thresh_deg: float = 5.0

    # Raw fixed-R PnP-Z is weakly observable here; keep this opt-in only.
    pnp_camera_z_correct_enable: bool = False
    pnp_camera_z_correct_min_frames: int = 2
    pnp_camera_z_correct_max_abs_m: float = 80.0
    pnp_camera_z_correct_max_std_m: float = 30.0
    pnp_camera_z_skip_dem_edges: bool = True
    camera_z_diagnostic_enable: bool = True

    # --- TTT speed knobs ---
    # Subset of DPT intermediate layers to treat as optimizable leaves in TTT.
    # None => use model.point_head.intermediate_layer_idx (typically [4,11,17,23]).
    # Example: [17, 23] — fewer leaves = smaller computation and fewer kernel
    # launches. Does NOT change math except for which tokens carry gradients.
    ttt_layers: Optional[List[int]] = None
    # Forward batch size for in-segment TTT main/overlap blocks.
    # 0 => one batch over all matched/overlap frames (fastest).
    # >0 => split into chunks of this size (use when VRAM-bound).
    ttt_batch: int = 0

    # If True, segment diagnostics include `pts3d_enu_full` (every frame,
    # full H×W pts3d in ENU as float32 numpy), `pts3d_conf_full`
    # (matching confidence), and `pts3d_frame_indices` (global frame indices).
    # Used by eval scripts to build a dense per-sequence point cloud for
    # F1 / PLY export. Off by default because it adds ~50MB CPU per segment
    # and runs an extra ENU transform on the full pts3d.
    export_full_pts3d: bool = False
