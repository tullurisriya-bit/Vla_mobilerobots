#!/usr/bin/env python3
"""
mppi_safety_node_working.py
----------------------------
V2: OmniVLA + MPPI safety layer + ArUco goal stopping

CHANGES FROM ORIGINAL:
  1. TF2 transform: obstacle points transformed from base_link to odom_lidar
  2. Odometry subscriber: updates self.state with real x,y,theta from kiss-icp
  3. Publishes /mppi/action_seq for viz node
  4. GOAL_RADIUS_M = 0.6

Run:
  Terminal 1: pixi run bash -c "source install/setup.bash && ros2 launch kiss_icp odometry.launch.py topic:=/velodyne_points visualize:=false"
  Terminal 2: pixi run python mppi_safety_node_working.py
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
from pi_mpc.mppi import MPPI

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
NUM_SAMPLES = 1000
V_MAX       = 0.5
W_MAX       = 1.0

OBSTACLE_COST_WEIGHT = 12.0
GOAL_COST_WEIGHT     = 1.5
OBSTACLE_RADIUS      = 0.6

AXES_SCALE           = 3.0
OMNIVLA_TIMEOUT      = 0.5
# ══════════════════════════════════════════════════════════════


class MPPISafetyNode(Node):

    def __init__(self):
        super().__init__('mppi_safety_node')

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

        # CHANGE 1: TF2 buffer and listener
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

        self.controller = MPPI(
            horizon     = HORIZON,
            num_samples = NUM_SAMPLES,
            dim_state   = 3,
            dim_control = 2,
            dynamics    = self._wheelchair_dynamics,
            cost_func   = self._cost_function,
            u_min       = torch.tensor([-V_MAX, -W_MAX]),
            u_max       = torch.tensor([ V_MAX,  W_MAX]),
            sigmas      = torch.tensor([0.2, 0.3]),
            lambda_     = 1.0,
        )

        self.create_subscription(Joy,         '/omnivla/joy',                   self._omnivla_callback,    10)
        self.create_subscription(PointCloud2, '/velodyne_points',               self._pointcloud_callback, 10)
        self.create_subscription(Image,       '/camera/camera/color/image_raw', self._camera_callback,     10)
        # CHANGE 2: odometry
        self.create_subscription(Odometry,    '/kiss/odometry',                 self._odom_callback,       10)

        self.cmd_pub    = self.create_publisher(Joy, '/whill/controller/joy', 10)
        # CHANGE 3: action seq for viz
        self.action_pub = self.create_publisher(Joy, '/mppi/action_seq',      10)

        self.create_timer(DT, self._control_loop)
        self.get_logger().info('MPPI Safety Node ready — waiting for OmniVLA...')

    # CHANGE 2: real odometry updates self.state
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
        cost = torch.zeros(state.shape[0], device=self.device)
        if self.obstacle_points.shape[0] > 0:
            pos      = state[:, :2].unsqueeze(1)
            obs      = self.obstacle_points.unsqueeze(0)
            min_dist = torch.norm(pos - obs, dim=2).min(dim=1).values
            self.get_logger().info(f'Min dist: {min_dist.min().item():.2f}m', throttle_duration_sec=1.0)  # ADD THIS
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
                self.get_logger().info('OmniVLA connected — MPPI active!')

    def _pointcloud_callback(self, msg: PointCloud2):
        # CHANGE 1: transform points from base_link to odom_lidar
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
            # rotate + translate to odom_lidar frame
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

        cmd.axes = [
            float(np.clip(omega * AXES_SCALE, -1.0, 1.0)),
            float(np.clip(v     * AXES_SCALE, -1.0, 1.0))
        ]
        self.cmd_pub.publish(cmd)

        # CHANGE 3: publish action sequence for viz
        seq_msg = Joy()
        seq_msg.header.stamp = cmd.header.stamp
        seq_msg.axes = action_seq.detach().cpu().flatten().tolist()
        self.action_pub.publish(seq_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MPPISafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
