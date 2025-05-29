from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='steering_system_identification',
            executable='steering_system_id',
            name='steering_system_id',
            output='screen',
            remappings=[
                ('~/input/control_cmd', '/control/command/control_cmd'),
                ('~/input/steering_report', '/vehicle/status/steering_status'),
            ],
        )
    ]) 