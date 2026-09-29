from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GeoV3Config:
    """GeoTTT-v3 分段适配的配置对象。

    本配置同时兼容两套命名：geo_v3 原生名 与 geo_v2 风格的命令行语义，
    使既有的 eval 命令可以直接平移过来。字段按功能分组，每组前有中文小标题。

    阅读提示：很多以 ``_enable`` / ``_weight`` / ``_max_*`` 结尾的开关是为消融
    实验（ablation）预留的细粒度旋钮；注释中标"已弃用 / 忽略"的字段仅为向后
    兼容保留，改动它们不会影响当前主路径。
    """

    # === 分段切分参数 ===
    segment_length: int = 32      # 单段长度（帧）
    overlap: int = 4              # 相邻段重叠帧数
    warmup_frames: int = 2        # 段首预热帧数

    # === GeoV2 兼容接口（命令行语义平移用）===
    geo_ttt_v2_mode: str = "pose_ttt_token"   # TTT 模式
    geo_ttt_v2_steps: int = 3                 # TTT 迭代步数
    geo_ttt_v2_kv_lr: float = 1e-2            # KV 学习率
    geo_ttt_v2_reg_weight: float = 0.01       # 正则权重
    geo_ttt_v2_loss: str = "mse"              # 损失类型
    geo_consist_stride: int = 1               # 一致性采样步长
    geo_consist_max_corr: int = 500           # 一致性最大对应点数
    mse_weight: float = 1.0                   # MSE 损失权重
    heading_weight: float = 0.0               # 朝向损失权重
    dense_reproj_weight: float = 0.5          # 稠密重投影 XY 权重
    dense_reproj_z_weight: float = 0.1        # 稠密重投影 Z 权重
    dense_reproj_huber_xy: float = 5.0        # 稠密重投影 XY 的 Huber 阈值（m）
    dense_reproj_huber_z: float = 10.0        # 稠密重投影 Z 的 Huber 阈值（m）
    # Plan C：仅对路面地面点做像素级重投影残差。
    dense_reproj_pixel_weight: float = 0.0    # 像素级重投影权重（0 关闭）
    dense_reproj_pixel_huber: float = 5.0     # 像素级重投影 Huber 阈值（px）
    cam_token_weight: float = 10.0            # 相机 token 损失权重
    max_frames_for_loss: int = 8              # 参与损失计算的最大帧数

    # === 当前实现使用的优化别名 ===
    kv_lr: float = 1e-2           # KV 学习率（实际生效名）
    num_steps: int = 3            # TTT 步数（实际生效名）
    reg_weight: float = 0.01      # 正则权重（实际生效名）
    loss_type: str = "mse"        # 损失类型（实际生效名）

    # === 几何 / 锚定 ===
    anchor_stride: int = 1        # 锚点采样步长
    # R2 map-component ablations.  Disabling DEM input keeps the DOM/RoMa
    # 2D georeference but removes every terrain-height signal.  Disabling
    # semantic masks retains the DEM but uses all geometric correspondences.
    dem_input_enable: bool = True
    semantic_masks_enable: bool = True

    # === 重置 / 链接策略 ===
    hard_reset: bool = True            # 是否在段边界做 hard reset
    use_segment_overlap: bool = True   # 是否使用分段重叠
    chain_only_outputs: bool = True    # 是否仅通过段输出做链接（不复用内部状态）

    # === DOM 裁剪 ===
    dom_crop_padding_px: int = 500     # DOM 裁剪四周留白（像素）

    # === 重叠帧连续性损失（TTT 内循环）===
    overlap_weight: float = 1.0        # 重叠帧连续性损失权重

    # === 重叠先验（overlap prior）===
    # 当头部重叠帧能给出低残差的 ENU SRT 时，仅把该 SRT 当作新段起始的
    # bootstrap / 裁剪先验使用；它**不是**下游主变换、DOM 重拟合门、PGO 门或 Z 门。
    overlap_prior_bootstrap_enable: bool = True
    # 已弃用的兼容名；现在内部等价于 bootstrap enable。
    overlap_prior_primary_enable: bool = True
    overlap_prior_primary_max_residual_m: float = 2.0   # 作为先验所允许的最大残差（m）
    overlap_prior_primary_min_frames: int = 3           # 启用先验所需的最少重叠帧
    overlap_prior_restrict_strong_supervision: bool = False  # 是否限制强监督
    overlap_prior_consistency_max_xy_m: float = 25.0    # 一致性允许的最大 XY 偏差（m）
    overlap_prior_dom_refit_min_new_frames: int = 3     # DOM 重拟合所需最少新帧
    overlap_prior_dom_refit_gate_enable: bool = True    # 是否启用 DOM 重拟合门
    # 已弃用 / 忽略：重叠先验不得锁定 DOM 重拟合的尺度。
    overlap_prior_dom_refit_fixed_scale_fallback: bool = False
    overlap_prior_dom_refit_scale_ratio_min: float = 0.75   # DOM 重拟合尺度比下限
    overlap_prior_dom_refit_scale_ratio_max: float = 1.35   # DOM 重拟合尺度比上限
    overlap_prior_dom_refit_max_heading_deg: float = 20.0   # DOM 重拟合最大朝向差（度）
    overlap_prior_dom_refit_max_overlap_drift_m: float = 5.0  # DOM 重拟合最大重叠漂移（m）
    overlap_prior_cache_reliable_only: bool = False     # 是否仅缓存可靠观测
    overlap_prior_pose_from_cache: bool = True          # 位姿是否取自缓存
    overlap_prior_pose_min_frames: int = 1              # 位姿先验所需最少帧
    overlap_prior_gate_pgo: bool = False                # 是否用先验给 PGO 设门
    overlap_prior_pgo_max_omega_rad: float = 0.35       # PGO 门：最大旋转（rad）
    overlap_prior_pgo_max_overlap_drift_m: float = 5.0  # PGO 门：最大重叠漂移（m）
    overlap_prior_pgo_max_t_delta_m: float = 30.0       # PGO 门：最大平移变化（m）
    overlap_prior_gate_z_correct: bool = False          # 是否用先验给 Z 校正设门
    overlap_prior_z_correct_max_abs_m: float = 50.0     # Z 校正最大绝对量（m）
    overlap_prior_z_correct_max_std_m: float = 80.0     # Z 校正最大标准差（m）

    # === DEM-Z 校正（来自匹配到的路面 DEM 点）===
    # legacy 的 dem_z_correct_enable 是总开关；"直接段平移校正"与"全局 PGO 边导出"
    # 可拆开做消融。道路样本不足时可使用 building-mask 已过滤的非建筑匹配点。
    dem_z_correct_enable: bool = True              # DEM-Z 校正总开关
    dem_z_direct_correct_enable: bool = True       # 是否直接对段做 Z 平移校正
    dem_z_pgo_edges_enable: bool = True            # 是否导出 DEM-Z 的 PGO 边
    dem_z_camera_correct_enable: bool = False      # 是否对相机中心做 DEM-Z 校正
    dem_z_camera_pgo_edges_enable: bool = False    # 是否导出相机 DEM-Z 的 PGO 边
    dem_z_road_direct_correct_enable: bool = True  # 路面优先；可回退到已过滤非建筑点
    dem_z_diagnostic_only: bool = False            # 仅诊断不实际校正
    dem_z_correct_min_frames: int = 5              # 触发校正所需最少帧
    dem_z_correct_min_points: int = 50             # 触发校正所需最少点
    dem_z_correct_max_abs_m: float = 20.0          # 校正最大绝对量（m）
    dem_z_correct_max_std_m: float = 8.0           # 校正最大标准差（m）
    dem_z_correct_allow_nonbuilding_direct: bool = False  # 兼容字段；dom_points 使用下方统一回退开关

    # Plan A：把逐帧 DOM 监督（位姿 + 稠密重投影）限制为**仅重叠帧**
    # （每段头尾各 `overlap` 帧）。段中部不受这些监督约束——仅由 StreamVGGT
    # 前向 + 平滑 + 重叠连续性驱动。
    dom_supervision_overlap_only: bool = False

    # === 诊断 ===
    max_grad_norm: float = 0.0    # 梯度裁剪阈值（0 表示不裁剪）
    # Map Registration Only baseline: keep one-shot DOM/RoMa/DEM registration
    # and the resulting model-to-ENU transform, but do not adapt tokens or
    # optimize/recenter the registration.
    map_anchor_only: bool = False
    # Explicit switch for a true no-TTT ablation.  Do not emulate this by
    # setting num_steps=0: the legacy TTT loop intentionally runs at least once.
    ttt_enable: bool = True

    # === 段内 PGO（Method-A）===
    # 流程：先做 KV-only TTT，再用 scipy 最小二乘对 (omega, t, delta) 做 1 次段级求解
    # （log_s 冻结）。启用后将**完全取代** fused-Adam 路径。
    seg_pgo_enable: bool = False                   # 段内 PGO 总开关
    seg_pgo_w_dom_xy: float = 5.0                  # DOM XY 边权重
    seg_pgo_w_dom_z: float = 2.0                   # DOM Z 边权重
    seg_pgo_w_consist: float = 8.0                 # 重叠一致性边权重
    seg_pgo_use_overlap_consistency: bool = False  # 是否启用重叠一致性边
    seg_pgo_w_delta_reg: float = 2.0               # delta 正则权重
    seg_pgo_w_pose_rot: float = 50.0               # 位姿旋转先验权重
    seg_pgo_w_pose_trans: float = 2.0              # 位姿平移先验权重
    seg_pgo_yaw_only_rotation: bool = False        # 旋转是否仅限 yaw（绕 Z）
    seg_pgo_freeze_delta: bool = True              # 是否冻结逐点 delta
    seg_pgo_use_point_edges: bool = False          # 是否使用直接点边
    seg_pgo_w_point_xy: float = 0.5                # 点边 XY 权重
    seg_pgo_w_point_z: float = 0.1                 # 点边 Z 权重
    seg_pgo_max_iter: int = 20                     # 最大迭代数
    seg_pgo_delta_clip: float = 5.0                # delta 截断阈值（m）
    seg_pgo_verbose: bool = False                  # 是否打印详细日志
    seg_pgo_gate_on_overlap_bootstrap: bool = True # 是否以重叠 bootstrap 设门
    seg_pgo_bootstrap_max_omega_rad: float = 0.25  # bootstrap 门：最大旋转（rad）
    seg_pgo_bootstrap_max_t_delta_m: float = 15.0  # bootstrap 门：最大平移变化（m）
    seg_pgo_bootstrap_max_overlap_motion_m: float = 5.0   # bootstrap 门：最大重叠运动（m）
    seg_pgo_bootstrap_max_prev_overlap_xy_m: float = 5.0  # bootstrap 门：上段重叠最大 XY（m）

    # === 直接 DOM/DEM 点对应（dom_point edges）===
    # 从匹配到的 query 像素上采样直接的 DOM/DEM 点对应，按段存储；
    # 当 SegmentPGO / 全局 PGO 对应的点边权重启用时被消费。
    dom_point_edges_enable: bool = False       # 点边总开关
    dom_point_edges_road_only: bool = False    # 是否仅用路面点
    dom_point_edges_max: int = 512             # 每段最多点边数
    # dom_points-only 的 Z 回退：当路面 DEM-Z 样本不足时，允许直接 Z 校正和
    # 全局 PGO 使用已由 building mask 过滤的非建筑匹配点。
    dom_point_z_nonbuilding_fallback_enable: bool = False

    # --- dom_point bootstrap：用点对应为新段提供起始变换先验 ---
    dom_point_bootstrap_enable: bool = False                    # bootstrap 总开关
    dom_point_prefer_point_bootstrap_transform: bool = True     # 是否优先采用点 bootstrap 变换
    dom_point_bootstrap_require_overlap_consistency: bool = False  # 是否要求重叠一致性
    dom_point_bootstrap_min_overlap_consistency: int = 1        # 所需最少重叠一致帧
    # Optional planar-map stabilization. Full GeoXel keeps the legacy free
    # Sim(3) fit because this transform is also consumed by CameraHead poses.
    # Map-anchor-only explicitly enables fixed-R in __post_init__ below.
    dom_point_bootstrap_fixed_rotation_refit_enable: bool = False
    dom_point_bootstrap_keep_pose_orientation: bool = False     # 是否保留位姿朝向
    dom_point_bootstrap_max_points: int = 4096                  # bootstrap 最多用点数
    dom_point_bootstrap_anchor_align: bool = True               # 是否做锚点对齐
    dom_point_bootstrap_prefer_overlap_frames: bool = True      # 是否优先用重叠帧
    # 相机重定心（camera recenter）：用点证据把相机 XY 拉回
    dom_point_bootstrap_camera_recenter_enable: bool = True
    # fixed-R 已从同一批地图点联合重拟合 s/t；默认禁止 frame-center 再次覆盖该平移。
    dom_point_bootstrap_recenter_after_fixed_rotation_refit_enable: bool = False
    dom_point_bootstrap_camera_recenter_segment0_enable: bool = False  # segment-0 是否也重定心
    dom_point_bootstrap_camera_recenter_max_pitch_deg: float = 12.0    # 允许的最大俯仰角（度）
    dom_point_bootstrap_camera_recenter_min_frames: int = 3            # 所需最少帧
    dom_point_bootstrap_camera_recenter_inlier_m: float = 15.0         # 内点阈值（m）
    dom_point_bootstrap_camera_recenter_max_apply_xy_m: float = 80.0   # 最大施加 XY 校正（m）
    dom_point_bootstrap_camera_recenter_xy_weight: float = 1.0         # XY 校正权重
    # AGL-Z 重定心：用 above-ground-level 把相机 Z 拉回
    dom_point_bootstrap_agl_z_recenter_enable: bool = True
    dom_point_bootstrap_agl_z_weight: float = 1.0                      # AGL-Z 校正权重
    dom_point_bootstrap_agl_z_max_apply_m: float = 80.0               # 最大施加 Z 校正（m）
    dom_point_bootstrap_agl_z_inlier_m: float = 15.0                  # AGL-Z 内点阈值（m）
    # 已弃用兼容旋钮；上一段的重叠几何不作为尺度来源。
    dom_point_bootstrap_inherit_overlap_scale: bool = False

    # --- dom_point SRT：用点对应直接拟合一个 Sim3（SRT）覆盖段变换 ---
    dom_point_srt_enable: bool = False           # SRT 总开关
    dom_point_srt_rematch: bool = False          # 是否重新匹配后再拟合
    dom_point_srt_ransac_iters: int = 256        # RANSAC 迭代数
    dom_point_srt_min_inliers: int = 32          # 最少内点数
    dom_point_srt_min_ratio: float = 0.05        # 最少内点比例
    dom_point_srt_thresh_m: float = 15.0         # 内点距离阈值（m）
    dom_point_srt_z_weight: float = 0.25         # Z 方向权重
    dom_point_srt_scale_ratio_min: float = 0.5   # 尺度比下限
    dom_point_srt_scale_ratio_max: float = 2.0   # 尺度比上限
    dom_point_srt_max_rot_deg: float = 60.0      # 允许的最大旋转（度）

    # === PointSRT 作为 teacher 的 TTT ===
    # 用鲁棒的 DOM/DEM 点对应构造 token-space 监督，在解码出最终位姿/点之前先做适配。
    # 与 dom_point_srt_enable 不同：本路径**不直接覆盖**段变换，校正是通过
    # token 适配（token adaptation）被吸收进去的。
    dom_point_ttt_enable: bool = False               # 该 TTT 路径总开关
    dom_point_ttt_use_srt_teacher: bool = True       # 是否用 SRT 作为 teacher
    dom_point_ttt_raw_gate_enable: bool = True       # 是否对原始点对应设质量门
    dom_point_ttt_raw_max_median_xy_m: float = 30.0  # 原始门：XY 中位数上限（m）
    dom_point_ttt_raw_max_p90_xy_m: float = 80.0     # 原始门：XY 90 分位上限（m）
    dom_point_ttt_raw_min_inlier_ratio: float = 0.25 # 原始门：最少内点比例
    dom_point_ttt_pose_teacher_source: str = "srt"   # 位姿 teacher 来源
    dom_point_ttt_road_only: bool = False            # 是否仅用路面点
    dom_point_ttt_max_points: int = 2048             # 最多用点数
    dom_point_ttt_min_inliers: int = 64              # 最少内点数
    dom_point_ttt_min_ratio: float = 0.25            # 最少内点比例
    dom_point_ttt_max_median_xy_m: float = 3.0       # 接受门：XY 中位数上限（m）
    dom_point_ttt_w_point_xy: float = 0.25           # 点 XY 监督权重
    dom_point_ttt_w_point_z: float = 0.02            # 点 Z 监督权重
    dom_point_ttt_w_pose_xy: float = 2.0             # 位姿 XY 监督权重
    dom_point_ttt_w_pose_z: float = 0.0              # 位姿 Z 监督权重
    dom_point_ttt_point_huber_m: float = 5.0         # 点监督 Huber 阈值（m）
    dom_point_ttt_pose_huber_m: float = 10.0         # 位姿监督 Huber 阈值（m）
    dom_point_ttt_replace_dense: bool = True         # 是否替换稠密重投影监督
    dom_point_ttt_pose_all_frames: bool = True       # 位姿监督是否覆盖所有帧
    dom_point_ttt_disable_direct_srt: bool = True    # 是否禁用直接 SRT 覆盖
    # 仅 TTT 用的相机中心 teacher，不创建 PGO 相机一元边。
    # "fixedR_pnp" 用被接受的固定旋转 PnP 的 XY，并保持当前 Z。
    dom_point_ttt_camera_teacher_source: str = "dom_points"
    # dom_points 模式下：用于段级 / 全局 PGO 一元边的相机目标来源。
    # "dom_points" 保留由点导出的自一致目标。
    dom_point_camera_teacher_enable: bool = False           # 相机 teacher 总开关
    dom_point_camera_teacher_source: str = "dom_points"     # 相机目标来源
    dom_point_camera_teacher_pgo_enable: bool = False       # 是否导出相机 PGO 边
    dom_point_camera_teacher_min_points: int = 8            # 最少点数
    dom_point_camera_teacher_max_points_per_frame: int = 256  # 每帧最多点数
    dom_point_camera_teacher_point_ray_ransac_iters: int = 128   # point-ray RANSAC 迭代数
    dom_point_camera_teacher_point_ray_inlier_thresh_m: float = 6.0  # point-ray 内点阈值（m）
    dom_point_camera_teacher_point_ray_min_ratio: float = 0.25   # point-ray 最少内点比例
    dom_point_camera_teacher_point_ray_max_median_m: float = 4.0  # point-ray 中位残差上限（m）
    dom_point_camera_teacher_point_ray_max_p90_m: float = 10.0   # point-ray 90 分位上限（m）
    dom_point_camera_teacher_point_ray_max_prior_xy_m: float = 50.0  # 相对先验 XY 上限（m）
    dom_point_camera_teacher_point_ray_max_prior_z_m: float = 120.0  # 相对先验 Z 上限（m）
    dom_point_camera_teacher_point_ray_use_z: bool = False       # point-ray 是否用 Z
    dom_point_pose_crop_scale: float = 1.4                       # 位姿裁剪缩放系数

    # === dom_points-only 质量路由 ===
    # 好的 DOM 点 SRT 保留既有 point-ray / frame-center 路径；差的点 SRT 可回退到
    # 被接受的固定旋转 PnP 相机中心，差的点边则不导出。
    dom_point_adaptive_pnp_enable: bool = True            # 自适应 PnP 回退总开关
    dom_point_quality_gate_enable: bool = True            # 质量门总开关
    dom_point_quality_min_inlier_ratio: float = 0.65      # 质量门：最少内点比例
    dom_point_quality_max_median_xy_m: float = 5.0        # 质量门：XY 中位数上限（m）
    dom_point_quality_min_road_frame_ratio: float = 0.30  # 质量门：最少路面帧比例
    dom_point_pnp_fallback_min_frames: int = 3            # PnP 回退：最少帧
    dom_point_pnp_fallback_min_inliers: int = 128         # PnP 回退：最少内点
    dom_point_pnp_fallback_min_positive_ratio: float = 0.95   # PnP 回退：最少正深度比例
    dom_point_pnp_fallback_max_reproj_px: float = 30.0    # PnP 回退：最大重投影误差（px）
    dom_point_pnp_fallback_max_ray_residual_m: float = 2.0    # PnP 回退：最大射线残差（m）
    dom_point_pnp_fallback_max_prior_xy_m: float = 40.0   # PnP 回退：相对先验 XY 上限（m）
    dom_point_pnp_fallback_gate_prior_xy: bool = True   # PnP 回退是否以先验 XY 设门

    # === dom_points-only 相机轨迹约束（ground ray）===
    # DOM 画面中心是一个**地面点**而非相机中心：预测的光轴射线应当与地面相交于
    # frame_center_enu.xy。据此约束相机轨迹。
    dom_point_ground_ray_enable: bool = False           # ground-ray 总开关
    dom_point_ground_ray_ttt_enable: bool = False       # TTT 阶段是否启用
    dom_point_ground_ray_seg_pgo_enable: bool = False   # 段级 PGO 阶段是否启用
    dom_point_ground_ray_pgo_enable: bool = False       # 全局 PGO 阶段是否启用
    dom_point_ground_ray_ttt_w_xy: float = 2.0          # TTT：XY 权重
    dom_point_ground_ray_ttt_huber_m: float = 10.0      # TTT：Huber 阈值（m）
    dom_point_ground_ray_seg_pgo_w_xy: float = 2.0      # 段级 PGO：XY 权重
    dom_point_ground_ray_pgo_w_xy: float = 2.0          # 全局 PGO：XY 权重
    dom_point_ground_ray_min_abs_dir_z: float = 0.05    # 射线方向 z 分量的最小绝对值（防除零）

    # === dom_points-only 观测语义 ===
    # 失败的位姿裁剪只作诊断、不作目标观测；当重叠连续性已被推翻时，段 PGO 可以
    # 用绝对 DOM / 射线证据重新锚定（re-anchor）。
    dom_point_observation_validity_enable: bool = True       # 观测有效性判定总开关
    dom_point_overlap_cache_reliable_only: bool = True       # 重叠缓存是否仅保留可靠观测
    dom_point_bootstrap_filter_head_overlap: bool = True     # bootstrap 是否过滤头部重叠帧
    dom_point_seg_pgo_absolute_reanchor_enable: bool = False  # 段 PGO 绝对重锚总开关
    dom_point_seg_pgo_reanchor_max_cost_ratio: float = 0.50  # 重锚接受的最大代价比
    dom_point_seg_pgo_reanchor_min_points: int = 32          # 重锚所需最少点
    dom_point_seg_pgo_reanchor_min_ground_rays: int = 4      # 重锚所需最少地面射线

    # === 目标语义（geo target）===
    # 优先采用从 DOM 匹配解析出的"地面足迹位移"作为相机 XY 校正的权威来源——
    # 在俯仰 / 近平面几何下，它比直接相信固定 R 的 PnP 相机中心更可观测。
    geo_target_mode: str = "auto"                  # 目标模式（auto / ...）
    # Standalone PnP replacement branch.  Default False preserves all legacy
    # target selection; when enabled, the evaluator requests the explicit
    # pointmap_sim3 path and records its source in diagnostics.
    pointmap_camera_transform_enable: bool = False
    # Experimental A/B modes; 'off' preserves the existing PointMap path too.
    pointmap_consistency: str = "off"  # off | map_only | joint
    pointmap_camera_lr: float = 1e-2
    pointmap_reprojection_weight: float = 0.1
    prefer_ground_signal_targets: bool = True      # 是否优先地面信号目标
    ground_signal_max_delta_xy_m: float = 120.0    # 地面信号允许的最大 XY 位移（m）
    ground_signal_min_coverage: float = 0.01       # 地面信号最小覆盖率
    segment_target_robust_fit: bool = True         # 段目标是否用鲁棒拟合

    # 历史开关名（兼容保留）：它启用"位姿导出的 DOM 目标"，但除非显式打开
    # allow_raw_pnp_camera_targets，否则不再使用原始固定 R 的 PnP 相机中心。
    prefer_pnp_camera_targets: bool = True         # 是否优先 PnP 相机目标（实为位姿导出目标）
    allow_raw_pnp_camera_targets: bool = False     # 是否允许使用原始 PnP 相机中心

    # 已弃用兼容旋钮：旧的 pitch 门不再决定目标语义；当前生效开关是
    # prefer_pnp_camera_targets。
    use_pnp_gate: bool = False                     # 旧 PnP 门（已弃用）
    pnp_pitch_thresh_deg: float = 5.0              # 旧 PnP 门俯仰阈值（已弃用）

    # 原始固定 R 的 PnP-Z 在此场景弱可观测；保持默认关闭、仅按需开启。
    pnp_camera_z_correct_enable: bool = False      # PnP-Z 校正总开关
    pnp_camera_z_correct_min_frames: int = 2       # 最少帧
    pnp_camera_z_correct_max_abs_m: float = 80.0   # 最大绝对量（m）
    pnp_camera_z_correct_max_std_m: float = 30.0   # 最大标准差（m）
    pnp_camera_z_skip_dem_edges: bool = True       # 是否跳过 DEM 边
    camera_z_diagnostic_enable: bool = True        # 相机 Z 诊断开关

    # === TTT 速度旋钮 ===
    # TTT 对照只改变被优化的变量；地图观测、GeoV3 loss、步数及下游 PGO 共用。
    # token: 优化指定 DPT token 叶子；lora: 每段零初始化并优化 aggregator LoRA。
    ttt_adaptation_type: str = "token"
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_target_blocks: str = "all"       # all | frame | global
    lora_target_layers: str = "qkv"       # qkv | qkv_proj | all
    lora_block_indices: Optional[List[int]] = None
    lora_lr: float = 1e-4
    lora_gradient_checkpointing: bool = True
    # 把哪些 DPT 中间层当作 TTT 的可优化叶子（leaf）。
    # None => 使用 model.point_head.intermediate_layer_idx（通常为 [4,11,17,23]）。
    # 例如 [17, 23]：叶子越少 => 计算量越小、kernel 启动越少。
    # 除了"哪些 token 携带梯度"外，不改变任何数学计算。
    ttt_layers: Optional[List[int]] = None
    # 诊断模式：在冻结 token 上仅做一次反传，记录各候选 DPT 层的
    # 尺度归一化梯度，不执行 optimizer.step()。用于选择 TTT 层，不用于评测。
    ttt_gradient_probe: bool = False
    # 段内 TTT main / overlap 块的前向 batch 大小。
    # 0  => 对所有匹配 / 重叠帧一次性批处理（最快）。
    # >0 => 按此大小分块（受显存限制时用）。
    ttt_batch: int = 0

    # 若为 True，段诊断中会包含 `pts3d_enu_full`（每帧完整 H×W 的 ENU pts3d，
    # float32 numpy）、`pts3d_conf_full`（匹配置信度）与 `pts3d_frame_indices`
    # （全局帧下标）。eval 脚本据此构建逐序列稠密点云用于 F1 / PLY 导出。
    # 默认关闭，因为它每段增加约 50MB CPU 内存，并对完整 pts3d 多做一次 ENU 变换。
    export_full_pts3d: bool = False

    def __post_init__(self) -> None:
        """Enforce the strict map-registration-only control semantics.

        The control keeps the one-shot DOM/RoMa/DEM point bootstrap that
        estimates each segment's model-to-ENU transform.  It must not
        subsequently refine tokens, trajectories, camera centers, or heights.
        Keeping this policy in the config prevents a dataset preset or shared
        command-line option from silently changing the ablation semantics.
        """
        if not 0.0 <= float(self.dom_point_quality_min_road_frame_ratio) <= 1.0:
            raise ValueError(
                "dom_point_quality_min_road_frame_ratio must be in [0, 1], "
                f"got {self.dom_point_quality_min_road_frame_ratio}"
            )
        self.ttt_adaptation_type = str(self.ttt_adaptation_type).strip().lower()
        if self.ttt_adaptation_type not in {"token", "lora"}:
            raise ValueError(
                "ttt_adaptation_type must be 'token' or 'lora', "
                f"got {self.ttt_adaptation_type!r}"
            )
        if int(self.lora_rank) <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
        if float(self.lora_alpha) <= 0.0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if float(self.lora_lr) <= 0.0:
            raise ValueError(f"lora_lr must be positive, got {self.lora_lr}")
        if self.lora_target_blocks not in {"all", "frame", "global"}:
            raise ValueError(
                "lora_target_blocks must be one of all/frame/global, "
                f"got {self.lora_target_blocks!r}"
            )
        if self.lora_target_layers not in {"qkv", "qkv_proj", "all"}:
            raise ValueError(
                "lora_target_layers must be one of qkv/qkv_proj/all, "
                f"got {self.lora_target_layers!r}"
            )
        # The point-map camera-transform branch is a strict, opt-in PnP
        # replacement.  It obtains camera centers only by applying the same
        # window model-point -> ENU Sim(3) used for the reconstructed points.
        # Keep all legacy defaults untouched when this flag is false.
        if self.pointmap_camera_transform_enable:
            self.geo_target_mode = "pointmap_sim3"
            self.prefer_pnp_camera_targets = False
            self.allow_raw_pnp_camera_targets = False
            self.dom_point_adaptive_pnp_enable = False
            self.dom_point_camera_teacher_enable = False
            self.dom_point_camera_teacher_pgo_enable = False
            self.dom_point_bootstrap_enable = True
            self.dom_point_prefer_point_bootstrap_transform = True
            self.dom_point_srt_enable = False
            # These stages directly alter camera centers independently of S_k.
            self.dom_point_bootstrap_camera_recenter_enable = False
            self.dom_point_bootstrap_agl_z_recenter_enable = False
            self.dem_z_camera_correct_enable = False
            self.dem_z_camera_pgo_edges_enable = False
            self.pnp_camera_z_correct_enable = False

        if self.pointmap_consistency not in {"off", "map_only", "joint"}:
            raise ValueError("pointmap_consistency must be off, map_only, or joint")
        if self.pointmap_consistency != "off":
            if not self.pointmap_camera_transform_enable or self.map_anchor_only:
                raise ValueError("PointMap consistency requires the PointMap branch without map_anchor_only")
            if self.ttt_adaptation_type != "token" or self.ttt_gradient_probe:
                raise ValueError("PointMap consistency requires token adaptation without gradient probing")
            if not 0.0 < self.pointmap_camera_lr < float("inf"):
                raise ValueError("pointmap_camera_lr must be finite and positive")
            if not 0.0 < self.pointmap_reprojection_weight < float("inf"):
                raise ValueError("pointmap_reprojection_weight must be finite and positive")
            if self.pointmap_consistency == "joint" and (
                not self.ttt_enable or self.geo_ttt_v2_steps <= 0
                or not 0.0 < self.geo_ttt_v2_kv_lr < float("inf")
            ):
                raise ValueError("joint PointMap consistency requires enabled TTT with positive steps/lr")
            self.ttt_enable = self.pointmap_consistency == "joint"
            self.dom_point_ttt_enable = self.ttt_enable
            self.dom_point_ttt_pose_teacher_source = "none"
            # A/B isolates camera/point consistency from every post-fit correction.
            self.seg_pgo_enable = False
            self.dom_point_bootstrap_anchor_align = False
            self.dom_point_bootstrap_fixed_rotation_refit_enable = False
            self.dom_point_bootstrap_keep_pose_orientation = False
            self.dom_point_bootstrap_require_overlap_consistency = False
            self.dom_point_bootstrap_filter_head_overlap = False
            self.dom_point_bootstrap_prefer_overlap_frames = False
            self.dem_z_correct_enable = False
            self.dem_z_direct_correct_enable = False
            self.dem_z_pgo_edges_enable = False
            self.dom_point_ground_ray_ttt_enable = False
            self.dom_point_ground_ray_seg_pgo_enable = False
            self.dom_point_ground_ray_pgo_enable = False

        if not self.map_anchor_only:
            return
        self.ttt_enable = False
        self.seg_pgo_enable = False
        self.seg_pgo_use_overlap_consistency = False
        self.seg_pgo_use_point_edges = False
        self.dem_z_correct_enable = False
        self.dem_z_direct_correct_enable = False
        self.dem_z_camera_correct_enable = False
        self.dem_z_pgo_edges_enable = False
        self.dem_z_camera_pgo_edges_enable = False
        self.dom_point_ttt_enable = False
        self.dom_point_camera_teacher_enable = False
        self.dom_point_camera_teacher_pgo_enable = False
        self.dom_point_ground_ray_ttt_enable = False
        self.dom_point_ground_ray_seg_pgo_enable = False
        self.dom_point_ground_ray_pgo_enable = False
        # Direct point edges are consumed by TTT/PGO and are not part of the
        # registration-only output path.  The bootstrap itself is retained: it
        # is the deterministic map registration being measured by this control.
        self.dom_point_edges_enable = False
        mode = str(self.geo_target_mode).strip().lower().replace("-", "_")
        if mode in {"pointmap_sim3", "pointmap_camera", "pointmap_transform"}:
            mode = "dom_points"
        self.dom_point_bootstrap_enable = mode in {
            "dom_points", "dompoints", "point_edges", "points",
        }
        self.dom_point_prefer_point_bootstrap_transform = True
        self.dom_point_bootstrap_fixed_rotation_refit_enable = True
        self.dom_point_bootstrap_camera_recenter_enable = False
        self.dom_point_bootstrap_agl_z_recenter_enable = False
        self.dom_point_srt_enable = False
