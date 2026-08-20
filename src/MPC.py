#!/usr/bin/env python3
"""
mppi_safety_node_v5_mpc.py
----------------------------
V5: OmniVLA + gradient-based MPC safety layer + ArUco goal stopping

This is the THIRD variant for comparison against the two MPPI variants
(V3 = pure MPPI, V4 = MPPI+CBF). NO CBF here -- per discussion, CBF and
MPPI/MPC were fighting each other in V4, so this variant is MPC alone,
directly comparable to V3 (also no CBF).

CHANGES FROM V3 (mppi_safety_node_v3.py / "pure MPPI, no CBF"):
  1. Core optimizer swapped: MPPI (random sampling + soft-argmin reweight)
     -> gradient-based MPC (iterative Adam gradient descent on a SINGLE
     trajectory, warm-started each control cycle from the previous
     solution, shifted forward by one step -- standard receding-horizon
     MPC warm-starting).
  2. self._cost_function is 100% UNCHANGED from V3 -- same obstacle
     quadratic penalty, same weight, same radius, same goal-tracking term.
     Only HOW the optimal action sequence is found differs, not what
     "good" means.
  3. Everything else -- TF2 transform (base_link/velodyne -> odom_lidar),
     real odometry via kiss-icp, ArUco goal detection + stopping, OmniVLA
     timeout handling, /whill/controller/joy and /mppi/action_seq
     publishers -- is IDENTICAL to V3. No CBF anywhere in this file.

HONEST TRADEOFF vs MPPI (worth knowing when comparing results):
  MPPI explores many random candidate trajectories in parallel, so it is
  naturally robust to non-convex cost landscapes (e.g. an obstacle sitting
  symmetrically in the path, where it's genuinely ambiguous whether to go
  left or right) -- it can "try" both directions across different samples
  and pick whichever scores better.
  Gradient-based MPC instead follows the LOCAL gradient of a single
  trajectory. If the cost landscape has a local minimum or a symmetric
  ambiguity right where it starts, it can get stuck, oscillate, or commit
  early to a suboptimal side, since it never randomly explores an
  alternative. Warm-starting also means a bad early decision can persist
  across ticks rather than being explored away like MPPI would.
  This is a real, known, literature-recognized tradeoff (sampling-based
  vs. gradient-based MPC), not a bug -- worth reporting honestly if your
  trials show a difference in how the two variants handle a
  directly-in-path obstacle specifically.

Run:
  Terminal 1: pixi run bash -c "source install/setup.bash && ros2 launch kiss_icp odometry.launch.py topic:=/velodyne_points visualize:=false"
  Terminal 2: pixi run python mppi_safety_node_v5_mpc.py
  Terminal 3: pixi run python mppi_viz_node.py
"""

import time
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, PointCloud2, Image
from nav_msgs.msg import Odometry
from tf2_ros import Buffer, TransformListener
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch

# ══════════════════════════════════════════════════════════════
ARUCO_DICT        = cv2.aruco.DICT_5X5_50
ARUCO_MARKER_ID   = 0
ARUCO_MARKER_SIZE = 0.10
GOAL_RADIUS_M     = 0.8

FX = 386.8666076660156
FY = 386.3300476074219
CX = 325.23321533203125
CY = 239.91650390625

DIST_COEFFS = np.array([
    -0.05516780540347099,
     0.06615912914276123,
    -0.0006559304310940206,
     0.0004326182825025171,
    -0.021776093170046806
], dtype=np.float32)

DT          = 0.1
HORIZON     = 25
V_MAX       = 0.5
W_MAX       = 1.0

# SAME as V3 -- unchanged obstacle/goal cost weights
OBSTACLE_COST_WEIGHT = 12.0
GOAL_COST_WEIGHT     = 1.5
OBSTACLE_RADIUS      = 0.6

# MPC-specific: gradient optimizer settings (replaces MPPI's
# num_samples/sigmas/lambda_ -- these are the MPC equivalents of
# "how hard do we search each control cycle")
MPC_NUM_ITERS = 15    # gradient steps per control cycle
MPC_LR        = 0.05  # Adam learning rate on the control sequence

AXES_SCALE           = 3.0
OMNIVLA_TIMEOUT      = 0.5
# ══════════════════════════════════════════════════════════════


class GradientMPC:
    """Single-trajectory, gradient-based MPC via iterative Adam descent on
    the control sequence, with receding-horizon warm-starting. Drop-in
    replacement for the MPPI controller: call signature and return shape
    (action_seq, info) match pi_mpc.mppi.MPPI exactly, so the rest of the
    node (_control_loop) needs no changes at all."""

    def __init__(self, horizon, dim_control, dynamics, cost_func,
                 u_min, u_max, num_iters, lr, device):
        self.horizon = horizon
        self.dynamics = dynamics
        self.cost_func = cost_func
        self.num_iters = num_iters
        self.lr = lr
        self.device = device
        self.u_min = u_min.to(device)
        self.u_max = u_max.to(device)
        # warm-start buffer, carried across control cycles
        self.nominal = torch.zeros(horizon, dim_control, device=device)

    def __call__(self, state0):
        # warm start from previous cycle's (shifted) solution
        u = self.nominal.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([u], lr=self.lr)

        for _ in range(self.num_iters):
            optimizer.zero_grad()

            # roll out ONE trajectory (batch size 1) through the full horizon
            s = state0.unsqueeze(0)  # (1,3)
            states = []
            for t in range(self.horizon):
                a = u[t].unsqueeze(0)  # (1,2)
                s = self.dynamics(s, a)
                states.append(s)
            states_tensor = torch.cat(states, dim=0)  # (horizon,3)

            # cost_func is purely pointwise over its batch dim (no cross-row
            # coupling), so it's safe to reuse UNCHANGED here with the
            # "batch" dim representing horizon timesteps of one trajectory,
            # instead of MPPI's K parallel samples at one timestep.
            total_cost = self.cost_func(states_tensor, u, {}).sum()
            total_cost.backward()
            optimizer.step()

            with torch.no_grad():
                u.clamp_(min=self.u_min, max=self.u_max)

        solution = u.detach()

        # shift-warm-start: next cycle starts from this cycle's solution,
        # advanced by one step (standard receding-horizon MPC practice)
        self.nominal = torch.cat([solution[1:], solution[-1:].clone()], dim=0)

        return solution, {}


class MPCSafetyNode(Node):

    def __init__(self):
        super().__init__('mpc_safety_node')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Using device: {self.device}')

        self.state            = torch.zeros(3, device=self.device)
        self.omnivla_linear   = 0.0
        self.omnivla_angular  = 0.0
        self.obstacle_points  = torch.zeros((0, 2), device=self.device)
        self.goal_reached     = False
        self.omnivla_received = False
        self.last_omnivla_t   = None
        self.bridge           = CvBridge()

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.camera_matrix = np.array([
            [FX,  0, CX],
            [ 0, FY, CY],
            [ 0,  0,  1]
        ], dtype=np.float32)
        self.dist_coeffs = DIST_COEFFS

        # V5 CHANGE: GradientMPC instead of MPPI. Same u_min/u_max bounds,
        # same dynamics, same (unchanged) cost function.
        self.controller = GradientMPC(
            horizon      = HORIZON,
            dim_control  = 2,
            dynamics     = self._wheelchair_dynamics,
            cost_func    = self._cost_function,
            u_min        = torch.tensor([-V_MAX, -W_MAX]),
            u_max        = torch.tensor([ V_MAX,  W_MAX]),
            num_iters    = MPC_NUM_ITERS,
            lr           = MPC_LR,
            device       = self.device,
        )

        self.create_subscription(Joy,         '/omnivla/joy',                   self._omnivla_callback,    10)
        self.create_subscription(PointCloud2, '/velodyne_points',               self._pointcloud_callback, 10)
        self.create_subscription(Image,       '/camera/camera/color/image_raw', self._camera_callback,     10)
        self.create_subscription(Odometry,    '/kiss/odometry',                 self._odom_callback,       10)

        self.cmd_pub    = self.create_publisher(Joy, '/whill/controller/joy', 10)
        self.action_pub = self.create_publisher(Joy, '/mppi/action_seq',      10)

        self.create_timer(DT, self._control_loop)
        self.get_logger().info('MPC Safety Node (V5) ready — waiting for OmniVLA...')

    def _odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw  = math.atan2(siny, cosy)
        self.state = torch.tensor([x, y, yaw], dtype=torch.float32, device=self.device)

    def _wheelchair_dynamics(self, state, action):
        x, y, theta = state[:, 0], state[:, 1], state[:, 2]
        v, w = action[:, 0], action[:, 1]
        return torch.stack([
            x + v * torch.cos(theta) * DT,
            y + v * torch.sin(theta) * DT,
            theta + w * DT
        ], dim=1)

    def _cost_function(self, state, action, info):
        # UNCHANGED from V3 -- same obstacle avoidance mechanism, same
        # weights. Only the optimizer calling this function differs.
        cost = torch.zeros(state.shape[0], device=self.device)
        if self.obstacle_points.shape[0] > 0:
            pos      = state[:, :2].unsqueeze(1)
            obs      = self.obstacle_points.unsqueeze(0)
            min_dist = torch.norm(pos - obs, dim=2).min(dim=1).values
            cost    += OBSTACLE_COST_WEIGHT * torch.clamp(OBSTACLE_RADIUS - min_dist, min=0.0) ** 2
        cost += GOAL_COST_WEIGHT * (
            (action[:, 0] - self.omnivla_linear)  ** 2 +
            (action[:, 1] - self.omnivla_angular) ** 2 * 0.5
        )
        return cost

    def _omnivla_callback(self, msg: Joy):
        if len(msg.axes) >= 2:
            self.omnivla_angular = float(msg.axes[0]) / AXES_SCALE
            self.omnivla_linear  = float(msg.axes[1]) / AXES_SCALE
            self.last_omnivla_t  = time.time()
            if not self.omnivla_received:
                self.omnivla_received = True
                self.get_logger().info('OmniVLA connected — MPC active!')

    def _pointcloud_callback(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                'odom_lidar', 'velodyne',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            q  = tf.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw  = math.atan2(siny, cosy)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            transform_available = True
        except Exception:
            transform_available = False

        raw_points = [
            [x, y, z] for x, y, z in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True)
            if -0.1 < z < 1.5 and (x**2 + y**2) < 10.0
        ]

        if not raw_points:
            self.obstacle_points = torch.zeros((0, 2), device=self.device)
            return

        pts = np.array(raw_points)

        if transform_available:
            xs = pts[:, 0] * cos_y - pts[:, 1] * sin_y + tx
            ys = pts[:, 0] * sin_y + pts[:, 1] * cos_y + ty
            points_2d = np.stack([xs, ys], axis=1).tolist()
        else:
            points_2d = pts[:, :2].tolist()

        self.obstacle_points = torch.tensor(points_2d, dtype=torch.float32, device=self.device)

    def _camera_callback(self, msg: Image):
        if self.goal_reached:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'CvBridge: {e}', throttle_duration_sec=5.0)
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            return
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id != ARUCO_MARKER_ID:
                continue
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i:i+1], ARUCO_MARKER_SIZE, self.camera_matrix, self.dist_coeffs
            )
            distance = float(np.linalg.norm(tvecs[0]))
            self.get_logger().info(f'ArUco detected — distance: {distance:.3f}m')
            if distance < GOAL_RADIUS_M:
                self.get_logger().info(f'GOAL REACHED at {distance:.3f}m — stopping!')
                self.goal_reached = True

    def _control_loop(self):
        cmd = Joy()
        cmd.header.stamp = self.get_clock().now().to_msg()

        if not self.omnivla_received:
            self.get_logger().info('Waiting for OmniVLA...', throttle_duration_sec=5.0)
            return

        if self.last_omnivla_t is not None:
            elapsed = time.time() - self.last_omnivla_t
            if elapsed > OMNIVLA_TIMEOUT:
                self.get_logger().info(
                    f'OmniVLA timeout ({elapsed:.2f}s) — stopping!',
                    throttle_duration_sec=1.0
                )
                cmd.axes = [0.0, 0.0]
                self.cmd_pub.publish(cmd)
                return

        if self.goal_reached:
            cmd.axes = [0.0, 0.0]
            self.cmd_pub.publish(cmd)
            return

        action_seq, _ = self.controller(self.state)
        best_action   = action_seq[0]

        v     = float(best_action[0].clamp(-V_MAX, V_MAX))
        omega = float(best_action[1].clamp(-W_MAX, W_MAX))

        # NO CBF filter in this variant -- MPC's own cost function is the
        # only obstacle-avoidance mechanism, matching V3's "pure" setup.
        cmd.axes = [
            float(np.clip(omega * AXES_SCALE, -1.0, 1.0)),
            float(np.clip(v     * AXES_SCALE, -1.0, 1.0))
        ]
        self.cmd_pub.publish(cmd)

        seq_msg = Joy()
        seq_msg.header.stamp = cmd.header.stamp
        seq_msg.axes = action_seq.detach().cpu().flatten().tolist()
        self.action_pub.publish(seq_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPCSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
