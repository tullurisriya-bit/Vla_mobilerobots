# Copyright (c) 2024 WHILL, Inc.
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='whill_driver',
            namespace='',
            executable='whill',
            name='whill',
            parameters=[PathJoinSubstitution([FindPackageShare('whill_bringup'), 'config', 'params.yaml'])],
        ),
    ])
