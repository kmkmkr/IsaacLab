# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def body_position_in_env_frame(
    env: ManagerBasedEnv,
    body_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the requested robot body position in the environment frame.

    OpenArm-Bi's articulation root is attached to the center support pillar. Using environment-frame positions keeps
    the task layout intuitive for table-top pick-and-place and avoids overloading policy observations with the pillar
    offset.
    """

    robot: Articulation = env.scene[robot_cfg.name]
    body_idx = robot.data.body_names.index(body_name)
    return robot.data.body_pos_w[:, body_idx] - env.scene.env_origins


def relative_object_position(
    env: ManagerBasedEnv,
    body_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return the object position relative to a robot body in the environment frame."""

    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    body_pos = body_position_in_env_frame(env, body_name=body_name, robot_cfg=robot_cfg)
    return object_pos - body_pos


def relative_goal_position(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
) -> torch.Tensor:
    """Return the place target position relative to the manipulated object in the environment frame."""

    object_pos = env.scene[object_cfg.name].data.root_pos_w - env.scene.env_origins
    target_pos = env.scene[target_cfg.name].data.root_pos_w - env.scene.env_origins
    return target_pos - object_pos


def hand_object_distance(
    env: ManagerBasedRLEnv,
    std: float,
    body_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward reaching with the selected hand body using a tanh kernel."""

    object_pos = env.scene[object_cfg.name].data.root_pos_w
    hand_pos = env.scene[robot_cfg.name].data.body_pos_w[:, env.scene[robot_cfg.name].data.body_names.index(body_name)]
    distance = torch.linalg.vector_norm(object_pos - hand_pos, dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
) -> torch.Tensor:
    """Reward moving the object toward the place target once it has been lifted off the table."""

    object_asset: RigidObject = env.scene[object_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    distance = torch.linalg.vector_norm(object_asset.data.root_pos_w - target_asset.data.root_pos_w, dim=1)
    lifted = object_asset.data.root_pos_w[:, 2] > minimal_height
    return lifted * (1.0 - torch.tanh(distance / std))


def object_goal_distance_while_grasped(
    env: ManagerBasedRLEnv,
    std: float,
    left_contact_sensor_name: str = "right_gripper_left_contact",
    right_contact_sensor_name: str = "right_gripper_right_contact",
    contact_threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
) -> torch.Tensor:
    """Reward moving the object toward the goal while both gripper fingers maintain contact.

    This matches the current task intent more closely than a lift-gated reward: the agent is encouraged to
    pick the object and carry/reach it to the target, regardless of whether it exceeds a separate lift threshold.
    """

    object_asset: RigidObject = env.scene[object_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    distance = torch.linalg.vector_norm(object_asset.data.root_pos_w - target_asset.data.root_pos_w, dim=1)
    grasped = both_fingers_in_contact(
        env=env,
        left_contact_sensor_name=left_contact_sensor_name,
        right_contact_sensor_name=right_contact_sensor_name,
        contact_threshold=contact_threshold,
    )
    return grasped * (1.0 - torch.tanh(distance / std))


def torque_limit_sticking_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joints that hover near their torque limit and are actively clipped.

    The term stays near zero while the actuator has comfortable headroom, then ramps up once the applied effort
    exceeds ``threshold * limit``. We blend in a bounded clipping term so persistent saturation is penalized more
    strongly than simply using a large but feasible torque.
    """

    robot: Articulation = env.scene[asset_cfg.name]
    effort_limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids].clamp_min(1e-6)
    applied_ratio = torch.abs(robot.data.applied_torque[:, asset_cfg.joint_ids]) / effort_limits
    clipping_ratio = (
        torch.abs(
            robot.data.computed_torque[:, asset_cfg.joint_ids] - robot.data.applied_torque[:, asset_cfg.joint_ids]
        )
        / effort_limits
    ).clamp(max=1.0)

    near_limit = ((applied_ratio - threshold) / max(1.0 - threshold, 1e-6)).clamp(min=0.0, max=1.0)
    return torch.mean(near_limit * (1.0 + clipping_ratio), dim=1)


def both_fingers_in_contact(
    env: ManagerBasedRLEnv,
    left_contact_sensor_name: str = "right_gripper_left_contact",
    right_contact_sensor_name: str = "right_gripper_right_contact",
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Return whether both parallel gripper fingers are in contact with the manipulated object."""

    left_contact_sensor: ContactSensor = env.scene.sensors[left_contact_sensor_name]
    right_contact_sensor: ContactSensor = env.scene.sensors[right_contact_sensor_name]

    left_contact = left_contact_sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3)
    right_contact = right_contact_sensor.data.force_matrix_w.reshape(env.num_envs, -1, 3)

    left_contact_mag = torch.linalg.vector_norm(left_contact, dim=-1).amax(dim=1)
    right_contact_mag = torch.linalg.vector_norm(right_contact, dim=-1).amax(dim=1)
    return (left_contact_mag > contact_threshold) & (right_contact_mag > contact_threshold)


def object_grasped_at_goal(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    max_speed: float,
    left_contact_sensor_name: str = "right_gripper_left_contact",
    right_contact_sensor_name: str = "right_gripper_right_contact",
    contact_threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
) -> torch.Tensor:
    """Return whether the object is at the place target while both gripper fingers contact it."""

    object_asset: RigidObject = env.scene[object_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]

    object_pos = object_asset.data.root_pos_w - env.scene.env_origins
    target_pos = target_asset.data.root_pos_w - env.scene.env_origins
    xy_error = torch.linalg.vector_norm(object_pos[:, :2] - target_pos[:, :2], dim=1)
    z_error = torch.abs(object_pos[:, 2] - target_pos[:, 2])
    object_speed = torch.linalg.vector_norm(object_asset.data.root_vel_w[:, :3], dim=1)
    both_contacts = both_fingers_in_contact(
        env=env,
        left_contact_sensor_name=left_contact_sensor_name,
        right_contact_sensor_name=right_contact_sensor_name,
        contact_threshold=contact_threshold,
    )

    placed = xy_error < xy_threshold
    placed = torch.logical_and(placed, z_error < z_threshold)
    placed = torch.logical_and(placed, object_speed < max_speed)
    placed = torch.logical_and(placed, both_contacts)
    return placed


def object_grasped_at_goal_bonus(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    max_speed: float,
    left_contact_sensor_name: str = "right_gripper_left_contact",
    right_contact_sensor_name: str = "right_gripper_right_contact",
    contact_threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
) -> torch.Tensor:
    """Sparse success bonus for objects placed at the target while held by both gripper fingers."""

    return object_grasped_at_goal(
        env=env,
        xy_threshold=xy_threshold,
        z_threshold=z_threshold,
        max_speed=max_speed,
        left_contact_sensor_name=left_contact_sensor_name,
        right_contact_sensor_name=right_contact_sensor_name,
        contact_threshold=contact_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        target_cfg=target_cfg,
    ).to(torch.float32)


def object_released_at_goal(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    max_speed: float,
    open_threshold: float,
    hand_clearance_threshold: float,
    body_name: str,
    left_contact_sensor_name: str = "right_gripper_left_contact",
    right_contact_sensor_name: str = "right_gripper_right_contact",
    contact_threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
) -> torch.Tensor:
    """Backward-compatible wrapper around the current contact-based success check."""

    del open_threshold, hand_clearance_threshold, body_name
    return object_grasped_at_goal(
        env=env,
        xy_threshold=xy_threshold,
        z_threshold=z_threshold,
        max_speed=max_speed,
        left_contact_sensor_name=left_contact_sensor_name,
        right_contact_sensor_name=right_contact_sensor_name,
        contact_threshold=contact_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        target_cfg=target_cfg,
    )


def object_released_at_goal_bonus(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_threshold: float,
    max_speed: float,
    open_threshold: float,
    hand_clearance_threshold: float,
    body_name: str,
    left_contact_sensor_name: str = "right_gripper_left_contact",
    right_contact_sensor_name: str = "right_gripper_right_contact",
    contact_threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
) -> torch.Tensor:
    """Backward-compatible sparse success wrapper around the current contact-based success check."""

    del open_threshold, hand_clearance_threshold, body_name
    return object_grasped_at_goal_bonus(
        env=env,
        xy_threshold=xy_threshold,
        z_threshold=z_threshold,
        max_speed=max_speed,
        left_contact_sensor_name=left_contact_sensor_name,
        right_contact_sensor_name=right_contact_sensor_name,
        contact_threshold=contact_threshold,
        robot_cfg=robot_cfg,
        object_cfg=object_cfg,
        target_cfg=target_cfg,
    )


def reset_object_and_goal(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_pose_range: dict[str, tuple[float, float]],
    target_pose_range: dict[str, tuple[float, float]],
    min_separation: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("place_target"),
):
    """Reset the object and place target in table coordinates with a minimum XY separation."""

    object_asset: RigidObject = env.scene[object_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]

    object_root_states = object_asset.data.default_root_state[env_ids].clone()
    target_root_states = target_asset.data.default_root_state[env_ids].clone()

    object_pose_delta = _sample_pose_deltas(object_pose_range, len(env_ids), object_asset.device)
    target_pose_delta = _sample_pose_deltas(target_pose_range, len(env_ids), target_asset.device)

    invalid = torch.linalg.vector_norm(object_pose_delta[:, :2] - target_pose_delta[:, :2], dim=1) < min_separation
    attempts = 0
    while torch.any(invalid) and attempts < 16:
        target_pose_delta[invalid] = _sample_pose_deltas(
            target_pose_range, int(invalid.sum().item()), target_asset.device
        )
        invalid = torch.linalg.vector_norm(object_pose_delta[:, :2] - target_pose_delta[:, :2], dim=1) < min_separation
        attempts += 1

    object_positions, object_orientations = _compose_root_state(
        root_states=object_root_states,
        pose_delta=object_pose_delta,
        env_origins=env.scene.env_origins[env_ids],
    )
    target_positions, target_orientations = _compose_root_state(
        root_states=target_root_states,
        pose_delta=target_pose_delta,
        env_origins=env.scene.env_origins[env_ids],
    )

    zero_velocities = torch.zeros((len(env_ids), 6), device=object_asset.device)

    object_asset.write_root_pose_to_sim(torch.cat([object_positions, object_orientations], dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(zero_velocities, env_ids=env_ids)
    target_asset.write_root_pose_to_sim(torch.cat([target_positions, target_orientations], dim=-1), env_ids=env_ids)
    target_asset.write_root_velocity_to_sim(zero_velocities, env_ids=env_ids)


def _sample_pose_deltas(
    pose_range: dict[str, tuple[float, float]],
    num_samples: int,
    device: str | torch.device,
) -> torch.Tensor:
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=device)
    return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_samples, 6), device=device)


def _compose_root_state(
    root_states: torch.Tensor, pose_delta: torch.Tensor, env_origins: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = root_states[:, :3] + env_origins + pose_delta[:, :3]
    orientation_delta = math_utils.quat_from_euler_xyz(pose_delta[:, 3], pose_delta[:, 4], pose_delta[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)
    return positions, orientations
