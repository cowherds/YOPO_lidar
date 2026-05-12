import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

_DEFAULT_YOPO = str(Path(__file__).resolve().parents[5] / "YOPO")


def generate_launch_description():
    yopo_root = LaunchConfiguration("yopo_root")
    weight = LaunchConfiguration("weight")

    yopo_root_arg = DeclareLaunchArgument(
        "yopo_root",
        default_value=_DEFAULT_YOPO,
        description="Absolute path to the YOPO/ folder.",
    )
    weight_arg = DeclareLaunchArgument(
        "weight",
        default_value="saved/YOPO_1/epoch50.pth",
        description="Weights .pth path (relative to yopo_root unless absolute).",
    )

    sim_ctrl_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("so3_quadrotor_simulator"),
                "launch",
                "simulator_attitude_control.launch.py",
            )
        )
    )

    sensor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sensor_simulator"),
                "launch",
                "sensor_simulator.launch.py",
            )
        )
    )

    yopo_node = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            [
                "cd ",
                yopo_root,
                " && source /opt/ros/${ROS_DISTRO:-humble}/setup.bash",
                " && conda run -n yopo bash scripts/run_yopo_ros2.sh ",
                " --weight=",
                weight,
            ],
        ],
        output="screen",
        emulate_tty=False,
    )

    return LaunchDescription([
        yopo_root_arg,
        weight_arg,
        sim_ctrl_launch,
        sensor_launch,
        yopo_node,
    ])
