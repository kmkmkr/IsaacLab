# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

from isaaclab_assets.robots.openarm import OPENARM_BI_CFG

from . import mdp

PEDESTAL_SIZE = (0.24, 0.24, 0.133)
PEDESTAL_CENTER = (0.245, -0.16, 0.0665)
PEDESTAL_TOP_Z = PEDESTAL_CENTER[2] + 0.5 * PEDESTAL_SIZE[2]
CUBE_CENTER_Z = PEDESTAL_TOP_Z + 0.055
LIFTED_OBJECT_CENTER_Z = CUBE_CENTER_Z + 0.02
PLACE_TARGET_Z_SAMPLE_RANGE = (0.0, 0.02)
OPENARM_REAL_CONTROL_HZ = 750
POLICY_CONTROL_HZ = 50
OPENARM_REAL_DECIMATION = OPENARM_REAL_CONTROL_HZ // POLICY_CONTROL_HZ
OPENARM_REAL_ARM_STIFFNESS = {
    "openarm_(left|right)_joint1": 70.0,
    "openarm_(left|right)_joint2": 70.0,
    "openarm_(left|right)_joint3": 70.0,
    "openarm_(left|right)_joint4": 60.0,
    "openarm_(left|right)_joint5": 10.0,
    "openarm_(left|right)_joint6": 10.0,
    "openarm_(left|right)_joint7": 10.0,
}
OPENARM_REAL_ARM_DAMPING = {
    "openarm_(left|right)_joint1": 2.75,
    "openarm_(left|right)_joint2": 2.5,
    "openarm_(left|right)_joint3": 2.0,
    "openarm_(left|right)_joint4": 2.0,
    "openarm_(left|right)_joint5": 0.7,
    "openarm_(left|right)_joint6": 0.6,
    "openarm_(left|right)_joint7": 0.5,
}
OPENARM_GRIPPER_STIFFNESS = 2000.0
OPENARM_GRIPPER_DAMPING = 100.0


@configclass
class OpenArmBiPickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene for a right-arm-centric OpenArm-Bi cube pick-and-place task."""

    robot: ArticulationCfg = MISSING

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        # Keep the cube anchored near the right arm's nominal reach envelope and lifted onto the helper pedestal.
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.26, -0.18, CUBE_CENTER_Z], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # The target is represented in the environment frame instead of the robot root frame because OpenArm-Bi's root is
    # attached to the center support pillar, which is inconvenient for sampling intuitive table-top place goals.
    place_target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.235, -0.15, CUBE_CENTER_Z], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=sim_utils.SphereCfg(
            radius=0.018,
            rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.8, 0.2), opacity=0.35),
        ),
    )

    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        init_state=AssetBaseCfg.InitialStateCfg(pos=PEDESTAL_CENTER),
        spawn=sim_utils.CuboidCfg(
            size=PEDESTAL_SIZE,
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.66, 0.54), roughness=0.9),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    right_gripper_left_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_right_left_finger",
        update_period=0.0,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    right_gripper_right_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_right_right_finger",
        update_period=0.0,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


@configclass
class ActionsCfg:
    """Action specifications for the task."""

    right_arm_action: ActionTerm = MISSING
    right_gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Low-dimensional observations designed to match real-robot sensor availability."""

        right_arm_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_arm_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"])},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        right_gripper_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        right_gripper_joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_arm_joint_effort = ObsTerm(
            func=base_mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"])},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        right_gripper_joint_effort = ObsTerm(
            func=base_mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_finger_joint.*"])},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_position = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        place_target_position = ObsTerm(
            func=base_mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("place_target")},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        right_ee_tcp_position = ObsTerm(
            func=mdp.body_position_in_env_frame,
            params={"body_name": "openarm_right_ee_tcp"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        object_to_right_ee_tcp = ObsTerm(
            func=mdp.relative_object_position,
            params={"body_name": "openarm_right_ee_tcp"},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        object_to_place_target = ObsTerm(
            func=mdp.relative_goal_position,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset and randomization events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.075, 0.075),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"]),
        },
    )

    reset_object_and_goal = EventTerm(
        func=mdp.reset_object_and_goal,
        mode="reset",
        params={
            # Sample inside a conservative right-arm tabletop workspace so the cube is always reachable.
            "object_pose_range": {"x": (-0.03, 0.03), "y": (-0.04, 0.04), "z": (0.0, 0.0)},
            # Keep the goal's vertical randomization modest so the current release-based success condition
            # remains achievable without requiring the object to hover unsupported in mid-air.
            "target_pose_range": {"x": (-0.035, 0.035), "y": (-0.05, 0.05), "z": PLACE_TARGET_Z_SAMPLE_RANGE},
            "min_separation": 0.09,
            "object_cfg": SceneEntityCfg("object"),
            "target_cfg": SceneEntityCfg("place_target"),
        },
    )


@configclass
class RewardsCfg:
    """Dense rewards for staged pick-and-reach learning."""

    reaching_object = RewTerm(
        func=mdp.hand_object_distance,
        params={"std": 0.12, "body_name": "openarm_right_ee_tcp"},
        weight=0.75,
    )

    lifting_object = RewTerm(
        func=lift_mdp.object_is_lifted,
        params={"minimal_height": LIFTED_OBJECT_CENTER_Z},
        weight=4.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance_while_grasped,
        params={
            "std": 0.18,
            "left_contact_sensor_name": "right_gripper_left_contact",
            "right_contact_sensor_name": "right_gripper_right_contact",
            "contact_threshold": 1.0,
        },
        weight=10.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance_while_grasped,
        params={
            "std": 0.05,
            "left_contact_sensor_name": "right_gripper_left_contact",
            "right_contact_sensor_name": "right_gripper_right_contact",
            "contact_threshold": 1.0,
        },
        weight=6.0,
    )

    place_success = RewTerm(
        func=mdp.object_grasped_at_goal_bonus,
        params={
            "xy_threshold": 0.045,
            "z_threshold": 0.03,
            "max_speed": 0.15,
            "left_contact_sensor_name": "right_gripper_left_contact",
            "right_contact_sensor_name": "right_gripper_right_contact",
            "contact_threshold": 1.0,
        },
        weight=30.0,
    )

    torque_limit_sticking = RewTerm(
        func=mdp.torque_limit_sticking_penalty,
        params={
            "threshold": 0.85,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["openarm_right_joint.*"]),
        },
        weight=-0.2,
    )

    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=base_mdp.joint_vel_l2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["openarm_right_joint.*", "openarm_right_finger_joint.*"]
            )
        },
        weight=-1e-4,
    )


@configclass
class TerminationsCfg:
    """Task terminations."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=base_mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

    success = DoneTerm(
        func=mdp.object_grasped_at_goal,
        params={
            "xy_threshold": 0.045,
            "z_threshold": 0.03,
            "max_speed": 0.15,
            "left_contact_sensor_name": "right_gripper_left_contact",
            "right_contact_sensor_name": "right_gripper_right_contact",
            "contact_threshold": 1.0,
        },
    )


@configclass
class OpenArmBiPickPlaceCubeEnvCfg(ManagerBasedRLEnvCfg):
    """Simple low-dimensional pick-and-place task for OpenArm-Bi."""

    scene: OpenArmBiPickPlaceSceneCfg = OpenArmBiPickPlaceSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    commands = None
    curriculum = None

    # Utilities shared with the place MDP helpers.
    gripper_joint_names = ["openarm_right_finger_joint.*"]
    gripper_open_val = 0.044
    gripper_threshold = 0.008

    def __post_init__(self):
        # Match the real robot's low-level control rate while keeping the policy at 50 Hz.
        self.decimation = OPENARM_REAL_DECIMATION
        self.episode_length_s = 6.0

        self.sim.dt = 1.0 / OPENARM_REAL_CONTROL_HZ
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        robot_cfg = OPENARM_BI_CFG.copy()
        robot_cfg.init_state.joint_pos.update(
            {
                "openarm_left_joint.*": 0.0,
                "openarm_right_joint.*": 0.0,
                "openarm_left_finger_joint.*": 0.044,
                "openarm_right_finger_joint.*": 0.044,
            }
        )
        robot_cfg.actuators["openarm_arm"].stiffness = OPENARM_REAL_ARM_STIFFNESS
        robot_cfg.actuators["openarm_arm"].damping = OPENARM_REAL_ARM_DAMPING
        robot_cfg.actuators["openarm_gripper"].stiffness = OPENARM_GRIPPER_STIFFNESS
        robot_cfg.actuators["openarm_gripper"].damping = OPENARM_GRIPPER_DAMPING
        robot_cfg.spawn.activate_contact_sensors = True
        self.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.right_arm_action = base_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint.*"],
            scale=0.1,
        )
        self.actions.right_gripper_action = base_mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": self.gripper_open_val},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )


@configclass
class OpenArmBiPickPlaceCubeEnvCfg_PLAY(OpenArmBiPickPlaceCubeEnvCfg):
    """Smaller interactive variant for debugging and teleop checks."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # Tight tabletop framing for quick inspection of the grasp point and cube interaction.
        self.viewer.eye = (0.9, -0.75, 0.62)
        self.viewer.lookat = (0.24, -0.16, 0.20)
