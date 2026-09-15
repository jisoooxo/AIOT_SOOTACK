from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    param_file = PathJoinSubstitution([
        FindPackageShare('aiot_control_pkg'),
        'config',
        'aiot_5dof_current.yaml'
    ])

    enable_logger_arg = DeclareLaunchArgument(
        'enable_logger',
        default_value='true',
        description='Whether to run current_logger_node'
    )

    motor_interface_node = Node(
        package='aiot_control_pkg',
        executable='motor_interface_node',
        name='motor_interface_node',
        output='screen',
        parameters=[param_file]
    )

    current_control_node = Node(
        package='aiot_control_pkg',
        executable='aiot_current_control_node',
        name='aiot_current_control_node',
        output='screen',
        parameters=[param_file]
    )

    current_logger_node = Node(
        package='aiot_control_pkg',
        executable='current_logger_node',
        name='current_logger_node',
        output='screen',
        parameters=[param_file]
    )

    return LaunchDescription([
        enable_logger_arg,
        motor_interface_node,
        current_control_node,
        current_logger_node,
    ])
