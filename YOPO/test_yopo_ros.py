import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image

import cv2
import time
import torch
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from config.config import cfg
from policy.yopo_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *
from policy.range_image import pointcloud_to_range_image
from ros_compat import make_ros_adapter, import_point_cloud2, import_position_command

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        self.ros_version, self.ros = make_ros_adapter("yopo_net", force_version=self.config.get("ros_version"))
        print(f"YOPO using {self.ros_version.upper()} backend.")
        self._apply_default_topics()
        self.point_cloud2 = import_point_cloud2(self.ros_version)
        self.PositionCommand = import_position_command(self.ros_version)

        cfg["train"] = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.height = int(cfg['range_image_height'])
        self.width = int(cfg['range_image_width'])
        self.channels = int(cfg['range_image_channels'])
        self.lidar_vertical_fov = cfg['lidar_vertical_fov']
        self.lidar_sensing_horizon = cfg['lidar_sensing_horizon']
        self.min_dis = cfg.get('lidar_sensing_blind', 0.1)

        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']

        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance()
        self.traj_time = self.lattice_primitive.segment_time

        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.sensor_fps = max(float(self.config.get("sensor_fps", 10.0)), 1.0)

        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        self.lattice_traj_pub = self.ros.create_publisher(PointCloud2, "/yopo_net/lattice_trajs_visual", queue_size=1)
        self.best_traj_pub = self.ros.create_publisher(PointCloud2, "/yopo_net/best_traj_visual", queue_size=1)
        self.all_trajs_pub = self.ros.create_publisher(PointCloud2, "/yopo_net/trajs_visual", queue_size=1)
        self.ctrl_pub = self.ros.create_publisher(self.PositionCommand, self.config["ctrl_topic"], queue_size=1)
        self.range_image_pub = self.ros.create_publisher(Image, "/yopo_net/range_image", queue_size=1)

        self.odom_sub = self.ros.create_subscription(
            Odometry, self.config["odom_topic"], self.callback_odometry, queue_size=1)

        lidar_topic = self.config.get("lidar_topic", "/lidar_points")
        self.lidar_sub = self.ros.create_subscription(
            PointCloud2, lidar_topic, self.callback_lidar, queue_size=1)
        print(f"Subscribing to LiDAR topic: {lidar_topic}")

        self.goal_sub = self.ros.create_subscription(
            PoseStamped, "/move_base_simple/goal", self.callback_set_goal, queue_size=1)

        self.ros.sleep(1.0)
        self.timer_ctrl = self.ros.create_timer(self.ctrl_dt, self.control_pub)
        print(
            f"Topics | odom: {self.config['odom_topic']} | lidar: {self.config['lidar_topic']} | "
            f"ctrl: {self.config['ctrl_topic']}"
        )
        print("YOPO Net Node Ready! Sensor: LiDAR (range image)")
        self.ros.spin()

    def _apply_default_topics(self):
        default_topics = {
            "odom_topic": "/sim/odom",
            "lidar_topic": "/lidar_points",
            "ctrl_topic": "/so3_control/pos_cmd",
        }
        for key, value in default_topics.items():
            if not self.config.get(key):
                self.config[key] = value

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 2])
        self.arrive = False
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if np.linalg.norm(pos - self.goal) < 5 and not self.arrive:
            print("Arrive!")
            self.arrive = True

    def process_odom(self):
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        Rotation_wc = Rotation_wb
        Rotation_cw = Rotation_wc.T

        vel_w = self.desire_vel if self.plan_from_reference else np.array(
            [self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm, Rotation_wc

    def _pointcloud2_to_numpy(self, msg):
        fields = {f.name: (f.offset, f.datatype) for f in msg.fields}
        point_step = msg.point_step
        n_points = msg.width * msg.height
        data = np.frombuffer(msg.data, dtype=np.uint8)

        x_off = fields['x'][0]
        y_off = fields['y'][0]
        z_off = fields['z'][0]

        points = np.zeros((n_points, 3), dtype=np.float32)
        for i in range(n_points):
            offset = i * point_step
            points[i, 0] = np.frombuffer(data[offset + x_off:offset + x_off + 4], dtype=np.float32)[0]
            points[i, 1] = np.frombuffer(data[offset + y_off:offset + y_off + 4], dtype=np.float32)[0]
            points[i, 2] = np.frombuffer(data[offset + z_off:offset + z_off + 4], dtype=np.float32)[0]
        return points

    def _publish_range_image(self, range_image_chw):
        try:
            if self.range_image_pub.get_subscription_count() == 0:
                return
        except Exception:
            pass

        depth_channel = range_image_chw[0]
        depth_uint8 = (depth_channel * 255).clip(0, 255).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        msg = Image()
        msg.header.stamp = self.ros.now()
        msg.header.frame_id = 'body'
        msg.height = depth_bgr.shape[0]
        msg.width = depth_bgr.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.step = depth_bgr.strides[0]
        msg.data = depth_bgr.tobytes()
        self.range_image_pub.publish(msg)

    @torch.inference_mode()
    def callback_lidar(self, data):
        if not self.odom_init:
            return

        time0 = time.time()

        points = self._pointcloud2_to_numpy(data)
        valid = np.isfinite(points).all(axis=1)
        points = points[valid]

        fov_half = self.lidar_vertical_fov / 2.0
        range_image = pointcloud_to_range_image(
            points,
            vertical_lines=self.height,
            horizontal_num=self.width,
            vertical_fov_up=fov_half,
            vertical_fov_down=-fov_half,
            max_range=self.lidar_sensing_horizon,
            min_range=self.min_dis
        )
        range_image = range_image.transpose(2, 0, 1)[np.newaxis, :, :, :]

        self._publish_range_image(range_image[0])

        time1 = time.time()
        sensor_input = torch.from_numpy(range_image).to(self.device, non_blocking=True)
        obs_norm, Rotation_wc = self.process_odom()
        obs_input = self.state_transform.prepare_input(obs_norm.to(self.device, non_blocking=True))
        self.Rotation_wc = Rotation_wc

        time2 = time.time()
        endstate_pred, score_pred = self.policy(sensor_input, obs_input)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()

        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=self.visualize)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = np.argmin(score) if self.visualize else 0
        with self.lock:
            start_pos = self.desire_pos if self.plan_from_reference else np.array(
                (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array(
                (self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endstate_w[action_id, 0, 0] + start_pos[0],
                                              endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endstate_w[action_id, 1, 0] + start_pos[1],
                                              endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endstate_w[action_id, 2, 0] + start_pos[2],
                                              endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()

        self.print_time(time0, time1, time2, time3, time4, time5)

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
            return

        with self.lock:
            self.ctrl_time += self.ctrl_dt
            control_msg = self.PositionCommand()
            control_msg.header.stamp = self.ros.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array(
            (self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array(
            (self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        if self.ros.has_connections(self.best_traj_pub):
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = self.ros.now()
            header.frame_id = 'world'
            point_cloud_msg = self.point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        if self.visualize and self.ros.has_connections(self.lattice_traj_pub):
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = self.ros.now()
            header.frame_id = 'world'
            point_cloud_msg = self.point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        if self.visualize and self.ros.has_connections(self.all_trajs_pub):
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = self.ros.now()
            header.frame_id = 'world'
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            point_cloud_msg = self.point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def print_time(self, time0, time1, time2, time3, time4, time5):
        self.time_interpolation = self.time_interpolation + (time1 - time0)
        self.time_prepare = self.time_prepare + (time2 - time1)
        self.time_forward = self.time_forward + (time3 - time2)
        self.time_process = self.time_process + (time4 - time3)
        self.time_visualize = self.time_visualize + (time5 - time4)
        self.count = self.count + 1

        total_time = (time5 - time0) * 1000
        tolerance = 1000.0 / self.sensor_fps
        if total_time > tolerance:
            self.ros.logwarn(f"Warn: Processing time {(time5 - time0) * 1000:.2f} ms exceeds {tolerance:.2f} ms, may cause message lag!")
            print(f"\033[34mCurrent Time Consuming:\033[0m "
                  f"lidar-projection: \033[32m{1000 * (time1 - time0):.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * (time2 - time1):.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * (time3 - time2):.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * (time4 - time3):.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * (time5 - time4):.2f} ms\033[0m")
        if self.verbose or (total_time > tolerance):
            print(f"\033[34mAverage Time Consuming:\033[0m "
                  f"lidar-projection: \033[32m{1000 * self.time_interpolation / self.count:.2f} ms\033[0m; "
                  f"data-prepare: \033[32m{1000 * self.time_prepare / self.count:.2f} ms\033[0m; "
                  f"network-inference: \033[32m{1000 * self.time_forward / self.count:.2f} ms\033[0m; "
                  f"post-process: \033[32m{1000 * self.time_process / self.count:.2f} ms\033[0m; "
                  f"visualize-trajectory: \033[32m{1000 * self.time_visualize / self.count:.2f} ms\033[0m")

    def warm_up(self):
        dummy_input = torch.randn(1, self.channels, self.height, self.width).to(self.device)
        dummy_obs = torch.randn(1, 9, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num).to(self.device)
        for _ in range(10):
            self.policy(dummy_input, dummy_obs)
        print("Warm-up done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='weight/yopo_lidar.pth', help='model weight path')
    parser.add_argument('--use_tensorrt', type=bool, default=False)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--visualize', type=bool, default=True)
    parser.add_argument('--plan_from_reference', type=bool, default=True)
    parser.add_argument('--odom_topic', type=str, default=None)
    parser.add_argument('--lidar_topic', type=str, default=None)
    parser.add_argument('--ctrl_topic', type=str, default=None)
    parser.add_argument('--sensor_fps', type=float, default=10.0)
    parser.add_argument('--ros_version', type=str, default='ros2', help='must be ros2 (default)')
    parser.add_argument(
        '--sensor_mode',
        type=str,
        default='lidar',
        choices=['lidar', 'auto'],
        help='Deprecated (LiDAR-only build): use lidar or auto; depth mode was removed.',
    )
    parser.add_argument('--goal', type=float, nargs=3, default=[10.0, 0.0, 2.0])
    args = parser.parse_args()

    settings = vars(args)
    weight = settings.pop('weight')
    settings.pop('sensor_mode', None)
    YopoNet(settings, weight)
