#!/usr/bin/env python3
"""gRPC policy server: trained openpi pi0 LBM checkpoint on the host.

Why this is self-contained
--------------------------
The upstream `lbm_policy_server.py` + `lbm_policy_conversions.py` depend on
`pydrake.math.RigidTransform`, but Drake doesn't publish wheels for Ubuntu
20.04 (glibc 2.31). We avoid that import chain entirely by:

  * implementing the four gRPC servicers (PolicyStep / PolicyReset /
    GetPolicyMetadata / Health) directly against the generated proto stubs,
  * parsing/encoding the `RigidTransform` proto field with plain numpy
    (it carries `translation: [3 doubles]` and `rotation: [9 doubles, row-major]`,
    so we never need Drake on the server side).

This module only depends on:
  - openpi (for pi0 inference)
  - grpc + grpc_workspace.proto (pure-Python stubs)
  - numpy, PIL, pyarrow

It pairs with `eval_lbm_parallel.sh`, which keeps this server in the openpi
`.venv` and runs the lbm_eval client (Drake) inside a Docker container.

State / action layout (matches LbmDataConfig + lbm_action_fields.yaml):
    [right xyz(3) | right rot_6d(6) | right gripper(1) |
     left  xyz(3) | left  rot_6d(6) | left  gripper(1)]   -> 20 dims
"""

# pyarrow first — see scripts/train.py for the segfault rationale.
import pyarrow  # noqa: F401

import argparse
import asyncio
import logging
import sys
import threading
import time
import uuid
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import grpc
import numpy as np

from grpc_workspace.proto import (
    GetPolicyMetadata_pb2,
    GetPolicyMetadata_pb2_grpc,
    PolicyReset_pb2,
    PolicyReset_pb2_grpc,
    PolicyStep_pb2,
    PolicyStep_pb2_grpc,
    health_pb2,
    health_pb2_grpc,
)


# ---------------------------------------------------------------------------
# 6D rotation <-> SO(3), pure numpy.
#
# IMPORTANT: LBM/vla_foundry datasets use a ROW-based 6D convention:
#   rot_6d = [R[0, :], R[1, :]]
# where R is a 3x3 rotation matrix.
# Keep this consistent with:
#   - vla_foundry.data.robotics.utils.{matrix_to_rot_6d,rot_6d_to_matrix}
#   - robot_gym.multiarm_spaces_conversions.{matrix_to_rotation_6d,rotation_6d_to_matrix}
# ---------------------------------------------------------------------------

def rot6d_to_matrix(r: np.ndarray) -> np.ndarray:
    """[6] -> [3, 3] rotation via Gram-Schmidt on the first two rows."""
    a1 = r[:3].astype(np.float64)
    a2 = r[3:].astype(np.float64)
    n1 = np.linalg.norm(a1)
    if n1 < 1e-9:
        a1 = np.array([1.0, 0.0, 0.0])
        n1 = 1.0
    b1 = a1 / n1
    a2_orth = a2 - (b1 @ a2) * b1
    n2 = np.linalg.norm(a2_orth)
    if n2 < 1e-9:
        helper = np.array([0.0, 1.0, 0.0]) if abs(b1[0]) > 0.5 else np.array([1.0, 0.0, 0.0])
        a2_orth = helper - (b1 @ helper) * b1
        n2 = np.linalg.norm(a2_orth)
    b2 = a2_orth / n2
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """[3, 3] -> [6]: first two rows stacked."""
    return R[:2, :].reshape(6).astype(np.float32)


def _validate_rot6d_convention() -> None:
    """Fail fast if rot6d helpers violate the row-based LBM convention."""
    # A non-symmetric rotation matrix so row-vs-column mixups are detectable.
    R = np.array(
        [
            [0.36, -0.48, 0.80],
            [0.80, 0.60, 0.00],
            [-0.48, 0.64, 0.60],
        ],
        dtype=np.float64,
    )
    v = matrix_to_rot6d(R)
    expected = R[:2, :].reshape(6)
    if not np.allclose(v, expected, atol=1e-8):
        raise RuntimeError("matrix_to_rot6d is not row-based [R[0,:], R[1,:]].")

    R2 = rot6d_to_matrix(v)
    if not np.allclose(R2, R, atol=1e-8):
        raise RuntimeError("rot6d_to_matrix round-trip failed for row-based convention.")


# ---------------------------------------------------------------------------
# Proto <-> numpy helpers (no pydrake).
# ---------------------------------------------------------------------------

# proto image dtype enum -> numpy dtype
_DTYPE_MAP = {
    PolicyStep_pb2.DTYPE_INT8: np.int8,
    PolicyStep_pb2.DTYPE_INT16: np.int16,
    PolicyStep_pb2.DTYPE_UINT8: np.uint8,
    PolicyStep_pb2.DTYPE_UINT16: np.uint16,
}


def _decode_rgb(image_msg) -> np.ndarray:
    """Proto `Image` (raw bytes only — we don't use compressed images here)."""
    if image_msg.compression != PolicyStep_pb2.NONE:
        raise NotImplementedError(
            f"Server only handles uncompressed RGB images (got compression={image_msg.compression})"
        )
    dtype = _DTYPE_MAP[image_msg.dtype]
    arr = np.frombuffer(image_msg.data, dtype=dtype).reshape(
        image_msg.height, image_msg.width, image_msg.channels
    )
    if arr.dtype != np.uint8:
        # Should be uint8 for pi0; but coerce just in case.
        arr = arr.astype(np.uint8)
    return arr


def _proto_translation(rt_msg) -> np.ndarray:
    return np.asarray(rt_msg.translation, dtype=np.float64)  # [3]


def _proto_rotation_matrix(rt_msg) -> np.ndarray:
    return np.asarray(rt_msg.rotation, dtype=np.float64).reshape(3, 3)  # [3,3] row-major


def _pose_to_state_vec(rt_msg) -> np.ndarray:
    """RigidTransform proto -> [9] (xyz + rot_6d), float32."""
    xyz = _proto_translation(rt_msg).astype(np.float32)
    R = _proto_rotation_matrix(rt_msg)
    return np.concatenate([xyz, matrix_to_rot6d(R)]).astype(np.float32)


def _action_chunk_to_pose_msg(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """[9] -> (translation [3], rotation [9, row-major])."""
    xyz = vec[:3].astype(np.float64)
    R = rot6d_to_matrix(vec[3:9])
    return xyz, R.flatten()


def _build_observation_inputs(
    obs_msg,
    cam_base: str,
    cam_left: str,
    cam_right: str,
) -> Dict[str, Any]:
    """Translate a `MultiarmObservation` proto into pi0 input dict."""
    # Per-arm pose / gripper, indexed by `robot_name` / `gripper_name`.
    actual = obs_msg.robot.actual
    poses = {p.robot_name: p.pose for p in actual.pose_status}
    grippers = {g.gripper_name: g.gripper_position for g in actual.gripper_status}

    if "right::panda" not in poses or "left::panda" not in poses:
        raise KeyError(
            f"Missing arm in observation. Have {sorted(poses.keys())}, "
            f"need 'right::panda' + 'left::panda'."
        )
    right_pose = _pose_to_state_vec(poses["right::panda"])  # [9]
    left_pose = _pose_to_state_vec(poses["left::panda"])
    right_grip = np.float32(grippers.get("right::panda_hand", 0.0)).reshape(1)
    left_grip = np.float32(grippers.get("left::panda_hand", 0.0)).reshape(1)
    state = np.concatenate([right_pose, right_grip, left_pose, left_grip]).astype(np.float32)

    cams = {ci.camera_serial: ci for ci in obs_msg.visuo}
    def _img(name: str) -> np.ndarray:
        if name not in cams:
            raise KeyError(
                f"Camera {name!r} not in observation.visuo. "
                f"Have {sorted(cams.keys())}."
            )
        cam = cams[name]
        if not cam.has_rgb:
            raise ValueError(f"Camera {name!r} carries no RGB image.")
        return _decode_rgb(cam.camera_rgb.image)

    return {
        "base_image": _img(cam_base),
        "left_wrist_image": _img(cam_left),
        "right_wrist_image": _img(cam_right),
        "state": state,
        # MultiarmObservation.use_language_instruction guards the field.
        "_language_instruction": obs_msg.language_instruction if obs_msg.use_language_instruction else "",
    }


def _action_to_response_msg(action: np.ndarray) -> "PolicyStep_pb2.PosesAndGrippers":
    """[20] action vec -> PosesAndGrippers proto."""
    pg = PolicyStep_pb2.PosesAndGrippers()

    def _add_pose(name: str, slice_: slice):
        xyz, rot_flat = _action_chunk_to_pose_msg(action[slice_])
        ps = pg.pose_status.add()
        ps.robot_name = name
        ps.pose.translation.extend(xyz.tolist())
        ps.pose.rotation.extend(rot_flat.tolist())

    def _add_grip(name: str, val: float):
        gs = pg.gripper_status.add()
        gs.gripper_name = name
        gs.gripper_position = float(val)

    _add_pose("right::panda", slice(0, 9))
    _add_grip("right::panda_hand", action[9])
    _add_pose("left::panda", slice(10, 19))
    _add_grip("left::panda_hand", action[19])
    return pg


# ---------------------------------------------------------------------------
# Pi0 LBM policy.
# ---------------------------------------------------------------------------

class Pi0LbmPolicy:
    """Holds the openpi pi0 model + per-client action queues."""

    def __init__(
        self,
        *,
        config_name: str,
        ckpt_dir: str,
        n_action_steps: int = 8,
        base_camera: str = "scene_left_0",
        left_wrist_camera: str = "wrist_left_plus",
        right_wrist_camera: str = "wrist_right_minus",
        default_prompt: Optional[str] = None,
    ):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        self.config_name = config_name
        self.ckpt_dir = str(ckpt_dir)
        self.n_action_steps = int(n_action_steps)
        self.cam_base = base_camera
        self.cam_left = left_wrist_camera
        self.cam_right = right_wrist_camera
        self.default_prompt = default_prompt

        train_cfg = _config.get_config(config_name)
        self.action_horizon = train_cfg.model.action_horizon
        if self.n_action_steps > self.action_horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) > model action_horizon "
                f"({self.action_horizon})."
            )

        logging.info(f"Loading openpi policy: config={config_name!r}, ckpt={ckpt_dir!r}")
        self._policy = _policy_config.create_trained_policy(
            train_cfg, ckpt_dir, default_prompt=default_prompt
        )
        logging.info("Policy loaded.")

        self._client_queues: Dict[uuid.UUID, Deque[np.ndarray]] = {}
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def reset(self, client_id: uuid.UUID, seed: Optional[int]) -> None:
        with self._lock:
            self._client_queues[client_id] = deque()

    def step(self, client_id: uuid.UUID, infer_input: Dict[str, Any]) -> np.ndarray:
        with self._lock:
            queue = self._client_queues.setdefault(client_id, deque())
        if not queue:
            self._refill_queue(client_id, infer_input)
            queue = self._client_queues[client_id]
        return queue.popleft()

    def _refill_queue(self, client_id: uuid.UUID, infer_input: Dict[str, Any]) -> None:
        prompt = infer_input.pop("_language_instruction", "") or self.default_prompt or "perform the task"
        infer_input["prompt"] = prompt
        out = self._policy.infer(infer_input)
        actions = np.asarray(out["actions"])  # (action_horizon, 20)
        if actions.ndim != 2 or actions.shape[1] != 20:
            raise RuntimeError(
                f"Unexpected action shape: {actions.shape}, expected (H, 20)"
            )
        keep = min(self.n_action_steps, actions.shape[0])
        with self._lock:
            self._client_queues[client_id] = deque(actions[:keep])

    # -- introspection ----------------------------------------------------

    def metadata(self) -> "GetPolicyMetadata_pb2.PolicyMetadata":
        m = GetPolicyMetadata_pb2.PolicyMetadata()
        m.name = f"pi0_lbm_lora({self.config_name})"
        m.skill_type = "Multi"
        m.checkpoint_path = self.ckpt_dir
        m.is_language_conditioned = True
        m.git_repo = "openpi"
        m.git_sha = "unknown"
        return m


# ---------------------------------------------------------------------------
# gRPC servicers (no pydrake!).
# ---------------------------------------------------------------------------

class _StepService(PolicyStep_pb2_grpc.PolicyStepServiceServicer):
    def __init__(
        self,
        policy: Pi0LbmPolicy,
        executor: ThreadPoolExecutor,
        cams: Tuple[str, str, str],
        debug_dump_dir: Optional[str] = None,
    ):
        self._policy = policy
        self._executor = executor
        self._cam_base, self._cam_left, self._cam_right = cams
        self._debug_dump_dir = debug_dump_dir
        self._debug_dumped = False  # one-shot per server run

    async def PolicyStep(self, request, context):
        try:
            client_id = uuid.UUID(request.client_identifier)
            infer_input = _build_observation_inputs(
                request.observation, self._cam_base, self._cam_left, self._cam_right
            )
            loop = asyncio.get_running_loop()
            action_vec = await loop.run_in_executor(
                self._executor, self._policy.step, client_id, infer_input
            )
            action_msg = _action_to_response_msg(action_vec)

            # On the very first step we serve, snapshot what was observed and
            # what we're sending back. Lets us diff against training data.
            if self._debug_dump_dir and not self._debug_dumped:
                self._debug_dumped = True
                self._dump_first_step(request.observation, infer_input, action_vec, action_msg)

            return PolicyStep_pb2.PolicyStepResponse(action=action_msg, success=True)
        except Exception as e:
            logging.exception("PolicyStep failed")
            return PolicyStep_pb2.PolicyStepResponse(
                action=PolicyStep_pb2.PosesAndGrippers(), success=False
            )

    def _dump_first_step(self, obs_msg, infer_input, action_vec, action_msg):
        import os, json as _json, pickle
        from PIL import Image as _PIL
        os.makedirs(self._debug_dump_dir, exist_ok=True)

        # Snapshot per-camera image stats + dump the actual image arrays.
        cam_summary = []
        for ci in obs_msg.visuo:
            arr = _decode_rgb(ci.camera_rgb.image) if ci.has_rgb else None
            if arr is not None:
                _PIL.fromarray(arr).save(os.path.join(self._debug_dump_dir, f"obs_{ci.camera_serial}.png"))
                cam_summary.append({
                    "camera_serial": ci.camera_serial,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "mean_rgb": [float(arr[..., c].mean()) for c in range(arr.shape[-1])],
                })

        # State / action breakdown.
        actual = obs_msg.robot.actual
        pose_summary = []
        for p in actual.pose_status:
            pose_summary.append({
                "robot_name": p.robot_name,
                "translation": list(p.pose.translation),
                "rotation_first_row": list(p.pose.rotation[:3]),
            })
        grip_summary = [{"name": g.gripper_name, "position": g.gripper_position} for g in actual.gripper_status]

        ret_pose = []
        for ps in action_msg.pose_status:
            ret_pose.append({
                "robot_name": ps.robot_name,
                "translation": list(ps.pose.translation),
                "rotation_first_row": list(ps.pose.rotation[:3]),
            })
        ret_grip = [{"name": gs.gripper_name, "position": gs.gripper_position} for gs in action_msg.gripper_status]

        snapshot = {
            "language_instruction": obs_msg.language_instruction if obs_msg.use_language_instruction else None,
            "use_language_instruction": obs_msg.use_language_instruction,
            "cameras": cam_summary,
            "actual_poses": pose_summary,
            "actual_grippers": grip_summary,
            "state_vec_first6": infer_input["state"][:6].tolist(),
            "state_vec_full": infer_input["state"].tolist(),
            "action_vec_full": action_vec.tolist(),
            "returned_poses": ret_pose,
            "returned_grippers": ret_grip,
        }
        with open(os.path.join(self._debug_dump_dir, "first_step.json"), "w") as f:
            _json.dump(snapshot, f, indent=2)
        logging.info(f"Wrote first-step debug dump to {self._debug_dump_dir}")


class _ResetService(PolicyReset_pb2_grpc.PolicyResetServiceServicer):
    def __init__(self, policy: Pi0LbmPolicy):
        self._policy = policy

    async def PolicyReset(self, request, context):
        try:
            client_id = uuid.UUID(request.client_identifier)
            self._policy.reset(client_id, request.seed)
            return PolicyReset_pb2.PolicyResetResponse(success=True)
        except Exception:
            logging.exception("PolicyReset failed")
            return PolicyReset_pb2.PolicyResetResponse(success=False)


class _MetaService(GetPolicyMetadata_pb2_grpc.GetPolicyMetadataServiceServicer):
    def __init__(self, policy: Pi0LbmPolicy):
        self._policy = policy

    async def GetPolicyMetadata(self, request, context):
        return GetPolicyMetadata_pb2.GetPolicyMetadataResponse(
            policy_metadata=self._policy.metadata(), success=True
        )


class _HealthService(health_pb2_grpc.HealthServicer):
    async def Check(self, request, context):
        return health_pb2.HealthCheckResponse(status=health_pb2.HealthCheckResponse.SERVING)


# ---------------------------------------------------------------------------
# Server bootstrap.
# ---------------------------------------------------------------------------

THIRTY_MB = 30 * 1024 * 1024


async def _serve(args):
    policy = Pi0LbmPolicy(
        config_name=args.config,
        ckpt_dir=args.ckpt_dir,
        n_action_steps=args.n_action_steps,
        base_camera=args.base_camera,
        left_wrist_camera=args.left_wrist_camera,
        right_wrist_camera=args.right_wrist_camera,
        default_prompt=args.default_prompt,
    )

    # JAX inference concurrency is configurable; default keeps one worker
    # per server process to avoid excessive compile / memory pressure.
    executor = ThreadPoolExecutor(
        max_workers=args.infer_workers,
        thread_name_prefix="pi0-infer",
    )

    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", THIRTY_MB),
        ("grpc.max_receive_message_length", THIRTY_MB),
    ])
    cams = (args.base_camera, args.left_wrist_camera, args.right_wrist_camera)
    PolicyStep_pb2_grpc.add_PolicyStepServiceServicer_to_server(
        _StepService(policy, executor, cams, debug_dump_dir=args.debug_dump_dir), server)
    PolicyReset_pb2_grpc.add_PolicyResetServiceServicer_to_server(
        _ResetService(policy), server)
    GetPolicyMetadata_pb2_grpc.add_GetPolicyMetadataServiceServicer_to_server(
        _MetaService(policy), server)
    health_pb2_grpc.add_HealthServicer_to_server(_HealthService(), server)
    server.add_insecure_port(args.server_uri)
    await server.start()
    # The eval shell greps for this exact line to know we're ready.
    print(f"Started Server loop on {args.server_uri}...", flush=True)
    try:
        await server.wait_for_termination()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server-uri", default="0.0.0.0:50051",
                        help="address:port of the gRPC server (default: %(default)s)")
    parser.add_argument("--config", default="pi0_lbm_lora",
                        help="openpi training config name (default: %(default)s)")
    parser.add_argument("--ckpt-dir", required=True,
                        help="Path to a step subdirectory under checkpoints/<config>/<exp>/")
    parser.add_argument("--n-action-steps", type=int, default=8,
                        help="How many actions to execute before re-inferring")
    parser.add_argument("--base-camera", default="scene_left_0")
    parser.add_argument("--left-wrist-camera", default="wrist_left_plus")
    parser.add_argument("--right-wrist-camera", default="wrist_right_minus")
    parser.add_argument("--default-prompt", default=None,
                        help="Fallback prompt when MultiarmObservation.language_instruction is empty")
    parser.add_argument(
        "--infer-workers",
        type=int,
        default=1,
        help="Inference worker threads per server process (default: %(default)s).",
    )
    parser.add_argument("--debug-dump-dir", default=None,
                        help="If set, dump the first observation+action to this directory "
                             "(useful when debugging unexpected eval behavior).")
    args = parser.parse_args()
    if args.infer_workers <= 0:
        raise ValueError("--infer-workers must be > 0")
    _validate_rot6d_convention()
    asyncio.run(_serve(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
