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
    """GeoTTT-v3 segment runner with hard reset and output-only chaining.

    This runner intentionally enforces:
    - no cross-segment reuse of optimization state
    - per-segment anchor re-initialization
    - chaining only through segment outputs
    """

    def __init__(self, config: Optional[GeoV3Config] = None):
        self.config = config or GeoV3Config()

    def _choose_anchor_enu(
        self,
        seg_id: int,
        seg_start: int,
        summaries: List[SegmentSummary],
        anchor_world_xyz: np.ndarray,
    ) -> np.ndarray:
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
        last_idx = seg_end - 1
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
        **kwargs,
    ) -> SegmentInput:
        return SegmentInput(
            segment_id=seg_id,
            start_frame=seg.start,
            end_frame=seg.end,
            frames=frames[seg.start:seg.end],
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
            prev_trajectory_global=None,  # deprecated; local+transform is authoritative
            prev_trajectory_local=prev_trajectory_local,
            prev_transform=prev_transform,
            metadata={**dict(kwargs), "save_vis_dir": save_vis_dir},
        )

    def _dedupe_by_frame_index(
        self,
        chain: GlobalChainState,
        num_frames: int,
    ) -> np.ndarray:
        """Assemble per-frame ENU trajectory from per-segment (local, T_k) pairs.

        chain.trajectories stores model-space trajectories; ENU is computed here
        in one pass at the very end, never stored between segments.
        """
        if not chain.trajectories:
            return np.zeros((0, 3), dtype=np.float32)

        frame_map: Dict[int, np.ndarray] = {}
        for summary, traj_local in zip(chain.summaries, chain.trajectories):
            if traj_local is None or len(traj_local) == 0:
                continue
            # Convert model-space trajectory to ENU using this segment's T_k
            transform = summary.transform
            if transform is not None:
                traj_t = torch.tensor(
                    np.asarray(traj_local, dtype=np.float32),
                    dtype=transform.s.dtype,
                ).to(transform.s.device)
                traj_enu = transform.model_to_world(traj_t).detach().cpu().numpy()
            else:
                # Fallback: treat stored values as ENU (should not normally happen)
                traj_enu = np.asarray(traj_local, dtype=np.float32)

            start = int(summary.start_frame)
            for local_idx, point in enumerate(traj_enu):
                frame_idx = start + local_idx
                if frame_idx < 0 or frame_idx >= num_frames:
                    continue
                frame_map[frame_idx] = np.asarray(point, dtype=np.float32)

        if not frame_map:
            return np.zeros((0, 3), dtype=np.float32)

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
        if save_vis_dir is None:
            return

        prev_traj = self._trajectory_enu_from_summary(prev_summary)
        curr_traj = self._trajectory_enu_from_summary(curr_summary)
        if prev_traj is None or curr_traj is None:
            return

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
            if not (np.isfinite(prev_xyz[:3]).all() and np.isfinite(curr_xyz[:3]).all()):
                continue

            delta = curr_xyz[:3] - prev_xyz[:3]
            delta_xy = float(np.linalg.norm(delta[:2]))
            delta_z = float(delta[2])
            delta_3d = float(np.linalg.norm(delta))
            shared_geo = int(global_frame in cache_dict)
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
        num_frames = len(frames)
        print(f"[GeoV3Runner] start run: num_frames={num_frames}, segment_length={self.config.segment_length}, overlap={self.config.overlap}")
        ranges = segment_sequence(
            num_frames=num_frames,
            segment_length=self.config.segment_length,
            overlap=self.config.overlap,
        )
        print(f"[GeoV3Runner] segments={len(ranges)}")
        chain = GlobalChainState(
            global_origin=np.array(gt_enu[0], copy=True) if len(gt_enu) else None
        )
        if not isinstance(kwargs.get("geo_observation_cache_global"), dict):
            kwargs["geo_observation_cache_global"] = {}
        kwargs.setdefault("cache_overlap_geo_observations", True)

        for seg_id, seg in enumerate(ranges):
            print(f"[GeoV3Runner] seg {seg_id}: start={seg.start}, end={seg.end}, len={seg.end-seg.start}")
            runtime = SegmentRuntimeState(
                segment_id=seg_id,
                start_frame=seg.start,
                end_frame=seg.end,
            )
            anchor_enu = self._choose_anchor_enu(seg_id, seg.start, chain.summaries, anchor_world_xyz)
            hard_reset_segment_state(runtime, anchor_enu=anchor_enu, anchor_frame_idx=0)

            prev_summary = chain.summaries[-1].to_prev_summary() if chain.summaries else None
            prev_traj_g = chain.trajectories[-1] if chain.trajectories else None  # now model-space
            prev_traj_m = chain.summaries[-1].trajectory_local if chain.summaries else None
            prev_transform = chain.summaries[-1].transform if chain.summaries else None
            prev_submap = chain.diagnostics[-1].get("submap_state") if chain.diagnostics else None

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
                prev_trajectory_global=None,  # deprecated
                prev_trajectory_local=prev_traj_m,
                prev_transform=prev_transform,
                save_vis_dir=save_vis_dir,
                **kwargs,
            )
            # For segment 0 only: inject GT first-frame rotation so T_0 uses real R
            if seg_id == 0 and gt_first_frame_rotation is not None:
                import numpy as _np
                gt_rot = _np.asarray(gt_first_frame_rotation, dtype=_np.float32)
                seg_input = seg_input.__class__(
                    **{k: v for k, v in seg_input.__dict__.items() if k != "gt_rotation"},
                    gt_rotation=gt_rot,
                )

            seg_out = segment_infer_fn(
                model=model,
                segment_input=seg_input,
                runtime_state=runtime,
                config=self.config,
            )
            print(f"[GeoV3Runner] seg {seg_id} done: traj_g={None if seg_out.trajectory_global is None else len(seg_out.trajectory_global)}")

            traj_g = seg_out.trajectory_global  # ENU, kept for boundary reporting only
            traj_m = seg_out.trajectory_local    # model-space, the canonical chain data
            diag = seg_out.diagnostics
            seg_transform = getattr(seg_out, "transform", None)
            # Count segments that triggered DOM-anchor fallback
            if diag.get("is_fallback", False):
                chain.fallback_count += 1
                print(f"[GeoV3Runner] seg {seg_id}: DOM-anchor fallback triggered"
                      f" (total fallbacks so far: {chain.fallback_count})")
            runtime.ttt_outputs["transform"] = seg_transform
            runtime.ttt_outputs["submap_state"] = getattr(seg_out, "submap_state", None)
            runtime.ttt_outputs["geo_cache"] = getattr(seg_out, "geo_cache", None)

            summary = self._build_summary(
                seg_id, seg.start, seg.end, diag, traj_g, traj_m
            )
            # Store T_k in summary so the next segment can use it without going through ENU
            summary.transform = seg_transform
            summary.post_ttt_overlap_m0 = diag.get("post_ttt_overlap_m0")
            chain.summaries.append(summary)
            if len(chain.summaries) >= 2:
                self._write_overlap_visual_check(
                    chain.summaries[-2],
                    summary,
                    save_vis_dir,
                    kwargs.get("geo_observation_cache_global"),
                )
            # Store model-space trajectory (not ENU); _dedupe_by_frame_index converts at the end
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

            # --- Per-segment cleanup: counter CUDA allocator fragmentation. ---
            # Without this, ms/frame grows monotonically across segments
            # (observed: 200f=3173ms -> 2000f=8054ms in airzoo batch).
            # Drop refs to large per-segment objects and force allocator to
            # release cached blocks before the next segment starts.
            del seg_input, seg_out, runtime
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not chain.trajectories:
            return np.zeros((0, 3), dtype=np.float32), chain

        print(f"[GeoV3Runner] finished: total_segments={len(chain.summaries)}"
              f" fallback_count={chain.fallback_count}")
        global_traj = self._dedupe_by_frame_index(chain, len(frames))
        return global_traj, chain
