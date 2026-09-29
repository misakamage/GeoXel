from typing import Any, Callable, Dict, List, Optional, Tuple
import csv
import gc
import os
import numpy as np
import torch

from .config import GeoV3Config
from .segment import segment_sequence
from .state import SegmentRuntimeState, SegmentSummary, GlobalChainState
from .reset import hard_reset_segment_state
from .io import SegmentInput, SegmentOutput


class GeoV3Runner:
    """GeoTTT-v3 分段调度器：hard reset + 仅通过输出链接（output-only chaining）。

    这是整个 geo_v3 的顶层主循环。它刻意强制三条约束：
    - 分段之间**不复用**任何优化状态；
    - 每段都重新初始化锚点（per-segment anchor re-init）；
    - 段与段之间**只通过段输出**（model-space 轨迹 + T_k）链接，不传递内部状态。

    典型用法：构造后调用 ``run(...)``，传入模型、帧序列、GT ENU 及单段推理函数
    ``segment_infer_fn``（通常是 defaults.default_segment_infer_fn）。
    """

    def __init__(self, config: Optional[GeoV3Config] = None):
        # 未显式传入配置时使用默认 GeoV3Config。
        self.config = config or GeoV3Config()

    def _choose_anchor_enu(
        self,
        seg_id: int,
        seg_start: int,
        summaries: List[SegmentSummary],
        anchor_world_xyz: np.ndarray,
    ) -> np.ndarray:
        """为当前段选定锚点 ENU，统一规整成形如 (3,) 的 float32 numpy 拷贝。

        兼容 torch 张量 / numpy / 多帧数组输入：若是张量先 detach 转 numpy，
        若是多帧数组则取第一帧。返回的是拷贝，避免后续就地修改污染外部数据。
        """
        if hasattr(anchor_world_xyz, "detach"):
            anchor = anchor_world_xyz.detach().cpu().numpy()
        else:
            anchor = np.asarray(anchor_world_xyz, dtype=np.float32)
        anchor = np.asarray(anchor, dtype=np.float32)
        if anchor.ndim > 1:
            anchor = anchor[0]
        return np.array(anchor, copy=True)

    def _build_summary(
        self,
        seg_id: int,
        seg_start: int,
        seg_end: int,
        diag: Dict[str, Any],
        trajectory_global: np.ndarray,
        trajectory_local: np.ndarray,
    ) -> SegmentSummary:
        """把单段推理的诊断信息 ``diag`` 整理成传给下一段的 SegmentSummary。

        关键点：把 diag 里的"具名字段"（位姿 / sim / 残差 / 边界等）抽取到 summary
        的对应属性，其余键统一塞进 ``metadata``（用集合差集排除已抽走的键，
        避免重复）。
        """
        last_idx = seg_end - 1
        # 最后一帧 ENU（用于边界报告）；空轨迹时为 None。
        last_enu = trajectory_global[-1].copy() if len(trajectory_global) else None
        return SegmentSummary(
            segment_id=seg_id,
            start_frame=seg_start,
            end_frame=seg_end,
            last_frame_idx=last_idx,
            last_enu=last_enu,
            last_pose=diag.get("last_pose"),
            final_sim2=diag.get("final_sim2"),
            final_sim3=diag.get("final_sim3"),
            scale=diag.get("scale"),
            residual=diag.get("residual"),
            inlier_ratio=diag.get("inlier_ratio"),
            anchor_count=int(diag.get("anchor_count", 0)),
            quality_score=diag.get("quality_score"),
            trajectory_global=trajectory_global,
            trajectory_local=trajectory_local,
            boundary_start_enu=diag.get("boundary_start_enu"),
            boundary_end_enu=diag.get("boundary_end_enu", last_enu),
            boundary_start_pose=diag.get("boundary_start_pose"),
            boundary_end_pose=diag.get("boundary_end_pose", diag.get("last_pose")),
            can_propagate=bool(diag.get("can_propagate", True)),
            fallback_reason=diag.get("fallback_reason"),
            overlap_len=int(diag.get("overlap_len", 0)),
            overlap_dom_targets_enu=diag.get("overlap_dom_targets_enu"),
            # 其余未被上面具名抽取的 diag 键，全部并入 metadata。
            metadata={k: v for k, v in diag.items() if k not in {
                "last_pose", "final_sim2", "final_sim3", "scale", "residual",
                "inlier_ratio", "anchor_count", "quality_score",
                "boundary_start_enu", "boundary_end_enu", "boundary_start_pose",
                "boundary_end_pose", "can_propagate", "fallback_reason", "overlap_len",
                "overlap_dom_targets_enu",
            }},
        )

    def build_segment_input(
        self,
        frames: List[dict],
        dom_image: Any,
        anchor_world_xyz: Any,
        project_fn: Any,
        get_world_xyz_fn: Any,
        heading_fn: Any,
        query_points: Any,
        image_paths: Any,
        inv_project_fn: Any,
        geo_elev: Any,
        dom_transform: Any,
        seg_id: int,
        seg: Any,
        save_vis_dir: Any = None,
        prev_summary: Any = None,
        prev_submap_state: Any = None,
        prev_trajectory_global: Any = None,
        prev_trajectory_local: Any = None,
        prev_transform: Any = None,
        frame_device: Any = None,
        **kwargs,
    ) -> SegmentInput:
        """把整段级别的输入裁剪 / 打包成当前段的 SegmentInput。

        注意 ``frames`` 会按 ``seg.start:seg.end`` 切片；其余多出来的关键字参数
        与 save_vis_dir 一并塞进 metadata，向单段推理函数透传。

        ``prev_trajectory_global`` 一律写死为 None：全局轨迹已弃用，
        以 model-space 局部轨迹 + T_k 为权威来源。
        """
        segment_frames = frames[seg.start:seg.end]
        if frame_device is not None:
            segment_frames = [
                {
                    **frame,
                    "img": frame["img"].to(frame_device, non_blocking=True),
                }
                for frame in segment_frames
            ]
        return SegmentInput(
            segment_id=seg_id,
            start_frame=seg.start,
            end_frame=seg.end,
            frames=segment_frames,   # 仅取本段帧；可按需从 CPU 搬到模型设备
            dom_image=dom_image,
            anchor_world_xyz=anchor_world_xyz,
            project_fn=project_fn,
            get_world_xyz_fn=get_world_xyz_fn,
            heading_fn=heading_fn,
            query_points=query_points,
            image_paths=image_paths,
            inv_project_fn=inv_project_fn,
            geo_elev=geo_elev,
            dom_transform=dom_transform,
            prev_summary=prev_summary,
            prev_submap_state=prev_submap_state,
            prev_trajectory_global=None,  # 已弃用；以 local + transform 为权威
            prev_trajectory_local=prev_trajectory_local,
            prev_transform=prev_transform,
            metadata={**dict(kwargs), "save_vis_dir": save_vis_dir},
        )

    def _dedupe_by_frame_index(
        self,
        chain: GlobalChainState,
        num_frames: int,
    ) -> np.ndarray:
        """由各段的 (model-space 轨迹, T_k) 在末尾一次性拼装出逐帧 ENU 轨迹。

        ``chain.trajectories`` 存的是 model-space 轨迹；ENU 仅在此处一次性换算，
        段间从不存 ENU。重叠帧会被后处理的段覆盖（后段写入同一 frame_idx），
        缺帧则用上一已知点填充（hold-last），完全无数据时填零。

        返回：形如 (num_frames, 3) 的 float32 ENU 轨迹。
        """
        if not chain.trajectories:
            return np.zeros((0, 3), dtype=np.float32)

        # 全局帧下标 -> ENU 位置；后写入的段会覆盖重叠帧。
        frame_map: Dict[int, np.ndarray] = {}
        for summary, traj_local in zip(chain.summaries, chain.trajectories):
            if traj_local is None or len(traj_local) == 0:
                continue
            # 用该段的 T_k 把 model-space 轨迹换算到 ENU。
            transform = summary.transform
            if transform is not None:
                traj_t = torch.tensor(
                    np.asarray(traj_local, dtype=np.float32),
                    dtype=transform.s.dtype,
                ).to(transform.s.device)
                traj_enu = transform.model_to_world(traj_t).detach().cpu().numpy()
            else:
                # 兜底：没有 transform 时把存储值当作 ENU（正常流程不应发生）。
                traj_enu = np.asarray(traj_local, dtype=np.float32)

            start = int(summary.start_frame)
            for local_idx, point in enumerate(traj_enu):
                frame_idx = start + local_idx
                if frame_idx < 0 or frame_idx >= num_frames:
                    continue
                frame_map[frame_idx] = np.asarray(point, dtype=np.float32)

        if not frame_map:
            return np.zeros((0, 3), dtype=np.float32)

        # 按帧序还原；缺帧用最近一次已知点填充（hold-last），开头无值则填零。
        ordered = []
        last_point = None
        for frame_idx in range(num_frames):
            if frame_idx in frame_map:
                last_point = frame_map[frame_idx]
                ordered.append(last_point)
            elif last_point is not None:
                ordered.append(last_point.copy())
            else:
                ordered.append(np.zeros(3, dtype=np.float32))

        return np.stack(ordered, axis=0).astype(np.float32)

    def _trajectory_enu_from_summary(self, summary: SegmentSummary) -> Optional[np.ndarray]:
        """从某段摘要尽力取出其 ENU 轨迹（优先现成 global，否则用 T_k 换算 local）。

        取值优先级：
        1. 直接用 summary.trajectory_global（若非空）；
        2. 否则取 model-space 的 trajectory_local，配合 transform 换算到 ENU；
        3. 若连 transform 都没有，则把 local 直接当 ENU 返回（兜底）。
        无可用轨迹时返回 None。
        """
        traj_global = getattr(summary, "trajectory_global", None)
        if traj_global is not None and len(traj_global) > 0:
            return np.asarray(traj_global, dtype=np.float64)

        traj_local = getattr(summary, "trajectory_local", None)
        transform = getattr(summary, "transform", None)
        if traj_local is None or len(traj_local) == 0:
            return None
        if transform is None:
            return np.asarray(traj_local, dtype=np.float64)

        traj_t = torch.tensor(
            np.asarray(traj_local, dtype=np.float32),
            dtype=transform.s.dtype,
        ).to(transform.s.device)
        return transform.model_to_world(traj_t).detach().cpu().numpy().astype(np.float64)

    def _write_overlap_visual_check(
        self,
        prev_summary: SegmentSummary,
        curr_summary: SegmentSummary,
        save_vis_dir: Any,
        geo_observation_cache_global: Any,
    ) -> None:
        """诊断输出：把相邻两段在重叠帧上的 ENU 位置差异落盘成 CSV / TXT。

        仅当传入 save_vis_dir 时才工作。对每个重叠的全局帧，计算两段给出的 ENU
        位置差（XY 距离、Z 差、3D 距离），并统计是否存在共享的 geo 观测，
        最后汇总均值 / 最大值。这只是评估用的可视化检查，**不影响轨迹本身**。
        """
        if save_vis_dir is None:
            return

        # 取两段各自的 ENU 轨迹；任一缺失则跳过。
        prev_traj = self._trajectory_enu_from_summary(prev_summary)
        curr_traj = self._trajectory_enu_from_summary(curr_summary)
        if prev_traj is None or curr_traj is None:
            return

        # 计算两段在全局帧空间的重叠区间 [overlap_start, overlap_end)。
        prev_start = int(prev_summary.start_frame)
        prev_end = int(prev_summary.end_frame)
        curr_start = int(curr_summary.start_frame)
        curr_end = int(curr_summary.end_frame)
        overlap_start = max(prev_start, curr_start)
        overlap_end = min(prev_end, curr_end)
        if overlap_end <= overlap_start:
            return

        prev_id = int(prev_summary.segment_id)
        curr_id = int(curr_summary.segment_id)
        pair_dir = os.path.join(str(save_vis_dir), "overlap_pairs", f"seg{prev_id:02d}_{curr_id:02d}")
        os.makedirs(pair_dir, exist_ok=True)

        rows: List[Dict[str, Any]] = []
        delta_xy_values: List[float] = []
        delta_z_values: List[float] = []
        delta_3d_values: List[float] = []
        shared_geo_count = 0
        cache_dict = geo_observation_cache_global if isinstance(geo_observation_cache_global, dict) else {}

        # 逐个重叠帧：把全局帧下标换算回两段各自的局部下标，比较 ENU 位置差。
        for global_frame in range(overlap_start, overlap_end):
            prev_local = global_frame - prev_start
            curr_local = global_frame - curr_start
            if prev_local < 0 or curr_local < 0:
                continue
            if prev_local >= len(prev_traj) or curr_local >= len(curr_traj):
                continue
            prev_xyz = np.asarray(prev_traj[prev_local], dtype=np.float64).reshape(-1)
            curr_xyz = np.asarray(curr_traj[curr_local], dtype=np.float64).reshape(-1)
            if prev_xyz.shape[0] < 3 or curr_xyz.shape[0] < 3:
                continue
            # 跳过含 NaN / Inf 的帧，避免污染统计量。
            if not (np.isfinite(prev_xyz[:3]).all() and np.isfinite(curr_xyz[:3]).all()):
                continue

            delta = curr_xyz[:3] - prev_xyz[:3]
            delta_xy = float(np.linalg.norm(delta[:2]))   # 水平面距离
            delta_z = float(delta[2])                      # 高度差（带符号）
            delta_3d = float(np.linalg.norm(delta))        # 3D 欧氏距离
            shared_geo = int(global_frame in cache_dict)   # 该帧是否有共享 geo 观测
            shared_geo_count += shared_geo
            delta_xy_values.append(delta_xy)
            delta_z_values.append(abs(delta_z))
            delta_3d_values.append(delta_3d)
            rows.append({
                "global_frame": int(global_frame),
                "prev_seg": prev_id,
                "prev_local": int(prev_local),
                "next_seg": curr_id,
                "next_local": int(curr_local),
                "prev_x": f"{float(prev_xyz[0]):.6f}",
                "prev_y": f"{float(prev_xyz[1]):.6f}",
                "prev_z": f"{float(prev_xyz[2]):.6f}",
                "next_x": f"{float(curr_xyz[0]):.6f}",
                "next_y": f"{float(curr_xyz[1]):.6f}",
                "next_z": f"{float(curr_xyz[2]):.6f}",
                "delta_xy_m": f"{delta_xy:.6f}",
                "delta_z_m": f"{delta_z:.6f}",
                "delta_3d_m": f"{delta_3d:.6f}",
                "shared_geo_observation": shared_geo,
            })

        if not rows:
            return

        # 明细写 CSV，汇总统计写 TXT。
        csv_path = os.path.join(pair_dir, "overlap_check.csv")
        txt_path = os.path.join(pair_dir, "overlap_check.txt")
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        mean_xy = float(np.mean(delta_xy_values))
        max_xy = float(np.max(delta_xy_values))
        mean_abs_z = float(np.mean(delta_z_values))
        max_abs_z = float(np.max(delta_z_values))
        mean_3d = float(np.mean(delta_3d_values))
        max_3d = float(np.max(delta_3d_values))
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"overlap_pair=seg{prev_id:02d}_{curr_id:02d}\n")
            f.write(f"global_frame_range=[{overlap_start},{overlap_end})\n")
            f.write(f"num_overlap_frames={len(rows)}\n")
            f.write(f"shared_geo_observation_frames={shared_geo_count}/{len(rows)}\n")
            f.write(f"mean_delta_xy_m={mean_xy:.6f}\n")
            f.write(f"max_delta_xy_m={max_xy:.6f}\n")
            f.write(f"mean_abs_delta_z_m={mean_abs_z:.6f}\n")
            f.write(f"max_abs_delta_z_m={max_abs_z:.6f}\n")
            f.write(f"mean_delta_3d_m={mean_3d:.6f}\n")
            f.write(f"max_delta_3d_m={max_3d:.6f}\n")

        print(
            f"[GeoV3][OverlapVis] seg {prev_id}->{curr_id}: wrote {csv_path} "
            f"n={len(rows)} shared_geo={shared_geo_count}/{len(rows)} "
            f"mean_xy={mean_xy:.3f}m max_xy={max_xy:.3f}m"
        )

    def run(
        self,
        model: Any,
        frames: List[dict],
        gt_enu: np.ndarray,
        segment_infer_fn: Callable[..., Tuple[np.ndarray, np.ndarray, Dict[str, Any]]],
        dom_image: Any = None,
        project_fn: Any = None,
        inv_project_fn: Any = None,
        anchor_world_xyz: Any = None,
        get_world_xyz_fn: Any = None,
        query_points: Any = None,
        heading_fn: Any = None,
        image_paths: Any = None,
        save_vis_dir: Any = None,
        gt_first_frame_rotation: Any = None,
        **kwargs,
    ) -> Tuple[np.ndarray, GlobalChainState]:
        """GeoTTT-v3 主循环：切分序列 -> 逐段 hard reset + 推理 -> 末尾拼装 ENU 轨迹。

        参数：
            model:            StreamVGGT 模型。
            frames:           整段序列的逐帧数据。
            gt_enu:           GT 的 ENU 轨迹（首帧用作全局原点）。
            segment_infer_fn: 单段推理函数（通常为 default_segment_infer_fn）。
            其余:             DOM / 投影 / 锚点等地理上下文，逐段透传。
            gt_first_frame_rotation: 仅 segment-0 用，把真实首帧旋转注入 T_0。

        返回：
            (global_traj, chain)：global_traj 为逐帧 ENU 轨迹 (N, 3)；
            chain 为 GlobalChainState，含各段摘要 / 轨迹 / 变换 / 诊断。
        """
        num_frames = len(frames)
        print(f"[GeoV3Runner] start run: num_frames={num_frames}, segment_length={self.config.segment_length}, overlap={self.config.overlap}")
        # 1) 把序列切成带重叠的分段区间。
        ranges = segment_sequence(
            num_frames=num_frames,
            segment_length=self.config.segment_length,
            overlap=self.config.overlap,
        )
        print(f"[GeoV3Runner] segments={len(ranges)}")
        # 全局链路状态：原点取 GT 首帧 ENU。
        chain = GlobalChainState(
            global_origin=np.array(gt_enu[0], copy=True) if len(gt_enu) else None
        )
        # 确保存在跨段共享的 geo 观测缓存，并默认开启重叠观测缓存。
        if not isinstance(kwargs.get("geo_observation_cache_global"), dict):
            kwargs["geo_observation_cache_global"] = {}
        kwargs.setdefault("cache_overlap_geo_observations", True)
        try:
            frame_device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            frame_device = None

        # 2) 逐段处理。
        for seg_id, seg in enumerate(ranges):
            print(f"[GeoV3Runner] seg {seg_id}: start={seg.start}, end={seg.end}, len={seg.end-seg.start}")
            # 2a) 新建并 hard reset 本段运行时状态（分段之间不复用任何状态）。
            runtime = SegmentRuntimeState(
                segment_id=seg_id,
                start_frame=seg.start,
                end_frame=seg.end,
            )
            anchor_enu = self._choose_anchor_enu(seg_id, seg.start, chain.summaries, anchor_world_xyz)
            hard_reset_segment_state(runtime, anchor_enu=anchor_enu, anchor_frame_idx=0)

            # 2b) 从上一段摘要取出传播所需的衔接信息（首段为 None）。
            prev_summary = chain.summaries[-1].to_prev_summary() if chain.summaries else None
            prev_traj_g = chain.trajectories[-1] if chain.trajectories else None  # 现为 model-space
            prev_traj_m = chain.summaries[-1].trajectory_local if chain.summaries else None
            prev_transform = chain.summaries[-1].transform if chain.summaries else None
            prev_submap = chain.diagnostics[-1].get("submap_state") if chain.diagnostics else None

            # 2c) 打包本段输入。
            seg_input = self.build_segment_input(
                frames=frames,
                dom_image=dom_image,
                anchor_world_xyz=anchor_world_xyz,
                project_fn=project_fn,
                inv_project_fn=inv_project_fn,
                get_world_xyz_fn=get_world_xyz_fn,
                heading_fn=heading_fn,
                query_points=query_points,
                image_paths=image_paths,
                seg_id=seg_id,
                seg=seg,
                prev_summary=prev_summary,
                prev_submap_state=prev_submap,
                prev_trajectory_global=None,  # 已弃用
                prev_trajectory_local=prev_traj_m,
                prev_transform=prev_transform,
                frame_device=frame_device,
                save_vis_dir=save_vis_dir,
                **kwargs,
            )
            # 仅 segment-0：注入 GT 首帧旋转，使 T_0 采用真实 R（而非单位阵）。
            # 由于 SegmentInput 是 dataclass，这里用"除 gt_rotation 外复制全部字段 + 覆盖
            # gt_rotation"的方式重建实例。
            if seg_id == 0 and gt_first_frame_rotation is not None:
                import numpy as _np
                gt_rot = _np.asarray(gt_first_frame_rotation, dtype=_np.float32)
                seg_input = seg_input.__class__(
                    **{k: v for k, v in seg_input.__dict__.items() if k != "gt_rotation"},
                    gt_rotation=gt_rot,
                )

            # 2d) 调用单段推理。
            seg_out = segment_infer_fn(
                model=model,
                segment_input=seg_input,
                runtime_state=runtime,
                config=self.config,
            )
            print(f"[GeoV3Runner] seg {seg_id} done: traj_g={None if seg_out.trajectory_global is None else len(seg_out.trajectory_global)}")

            traj_g = seg_out.trajectory_global  # ENU，仅用于边界报告
            traj_m = seg_out.trajectory_local    # model-space，跨段链接的权威数据
            diag = seg_out.diagnostics
            seg_transform = getattr(seg_out, "transform", None)
            # 统计触发了 DOM 锚点回退（fallback）的段数。
            if diag.get("is_fallback", False):
                chain.fallback_count += 1
                print(f"[GeoV3Runner] seg {seg_id}: DOM-anchor fallback triggered"
                      f" (total fallbacks so far: {chain.fallback_count})")
            runtime.ttt_outputs["transform"] = seg_transform
            runtime.ttt_outputs["submap_state"] = getattr(seg_out, "submap_state", None)
            runtime.ttt_outputs["geo_cache"] = getattr(seg_out, "geo_cache", None)

            # 2e) 生成摘要并入链；T_k 存进摘要，便于下一段不经 ENU 直接使用。
            summary = self._build_summary(
                seg_id, seg.start, seg.end, diag, traj_g, traj_m
            )
            summary.transform = seg_transform
            summary.post_ttt_overlap_m0 = diag.get("post_ttt_overlap_m0")
            chain.summaries.append(summary)
            # 自第二段起，输出相邻段重叠帧的诊断检查（若开启了可视化目录）。
            if len(chain.summaries) >= 2:
                self._write_overlap_visual_check(
                    chain.summaries[-2],
                    summary,
                    save_vis_dir,
                    kwargs.get("geo_observation_cache_global"),
                )
            # 存 model-space 轨迹（非 ENU）；ENU 由 _dedupe_by_frame_index 在最后换算。
            chain.trajectories.append(traj_m)
            chain.transforms.append(seg_transform)
            chain.diagnostics.append({
                "segment_id": seg_id,
                "start": seg.start,
                "end": seg.end,
                "anchor_enu": anchor_enu,
                "diag": diag,
                "prev_summary": prev_summary,
            })

            # 2f) 逐段清理：对抗 CUDA allocator 碎片化。
            # 若不做，ms/frame 会随段数单调增长
            #（实测 airzoo 批次：200f=3173ms -> 2000f=8054ms）。
            # 丢弃大块的逐段对象引用，并在进入下一段前强制释放缓存显存块。
            del seg_input, seg_out, runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not chain.trajectories:
            return np.zeros((0, 3), dtype=np.float32), chain

        # 3) 末尾一次性把各段 model-space 轨迹拼装成逐帧 ENU 轨迹。
        print(f"[GeoV3Runner] finished: total_segments={len(chain.summaries)}"
              f" fallback_count={chain.fallback_count}")
        global_traj = self._dedupe_by_frame_index(chain, len(frames))
        return global_traj, chain
