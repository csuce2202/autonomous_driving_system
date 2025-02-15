import numpy as np
import carla
from typing import Dict, List, Optional, Tuple
import queue
import weakref
import cv2
from dataclasses import dataclass
import threading
import logging
from autonomous_driving_system.config import SensorConfig


@dataclass
class SensorData:
    """传感器数据结构"""
    frame: int
    timestamp: float
    data: np.ndarray


class SensorManager:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, config: SensorConfig):
        self.world = world
        self.vehicle = vehicle
        self.config = config

        self._sensors = {}
        self._sensor_queues = {}
        self._lock = threading.Lock()
        self._is_alive = True

        # 初始化传感器
        try:
            self._init_rgb_camera()
            self._init_lidar()
        except Exception as e:
            self.destroy()
            raise e

    def destroy(self):
        """安全地销毁所有传感器"""
        with self._lock:
            if not self._is_alive:
                return

            for sensor_type, sensor in self._sensors.items():
                try:
                    if sensor is not None and sensor.is_alive:
                        sensor.stop()
                        sensor.destroy()
                except Exception as e:
                    print(f"Warning: Error destroying {sensor_type} sensor: {e}")

            self._sensors.clear()
            self._sensor_queues.clear()
            self._is_alive = False

    def is_alive(self):
        """检查传感器管理器是否仍然活动"""
        return self._is_alive and all(
            sensor is not None and sensor.is_alive
            for sensor in self._sensors.values()
        )

    def _init_rgb_camera(self):
        """初始化RGB相机"""
        if 'rgb' in self._sensors:
            return

        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(self.config.rgb_camera['width']))
        bp.set_attribute('image_size_y', str(self.config.rgb_camera['height']))
        bp.set_attribute('fov', str(self.config.rgb_camera['fov']))

        location = carla.Location(*self.config.rgb_camera['location'])
        rotation = carla.Rotation(*self.config.rgb_camera['rotation'])
        transform = carla.Transform(location, rotation)

        with self._lock:
            self._sensors['rgb'] = self.world.spawn_actor(
                bp, transform, attach_to=self.vehicle
            )
            self._sensor_queues['rgb'] = queue.Queue()
            self._sensors['rgb'].listen(
                lambda image: self._rgb_callback(
                    weakref.proxy(image),
                    self._sensor_queues['rgb']
                )
            )

    def _init_lidar(self):
        """初始化LiDAR"""
        if 'lidar' in self._sensors:
            return

        bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        bp.set_attribute('channels', str(self.config.lidar['channels']))
        bp.set_attribute('range', str(self.config.lidar['range']))
        bp.set_attribute('points_per_second', str(self.config.lidar['points_per_second']))
        bp.set_attribute('rotation_frequency', str(self.config.lidar['rotation_frequency']))
        bp.set_attribute('upper_fov', str(self.config.lidar['upper_fov']))
        bp.set_attribute('lower_fov', str(self.config.lidar['lower_fov']))

        location = carla.Location(*self.config.lidar['location'])
        rotation = carla.Rotation(*self.config.lidar['rotation'])
        transform = carla.Transform(location, rotation)

        with self._lock:
            self._sensors['lidar'] = self.world.spawn_actor(
                bp, transform, attach_to=self.vehicle
            )
            self._sensor_queues['lidar'] = queue.Queue()
            self._sensors['lidar'].listen(
                lambda point_cloud: self._lidar_callback(
                    weakref.proxy(point_cloud),
                    self._sensor_queues['lidar']
                )
            )

    def _rgb_callback(self, image, sensor_queue):
        """RGB相机回调函数"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 去除alpha通道

        sensor_data = SensorData(
            frame=image.frame,
            timestamp=image.timestamp,
            data=array
        )
        sensor_queue.put(sensor_data)

    def _lidar_callback(self, point_cloud, sensor_queue):
        """LiDAR回调函数"""
        points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (-1, 4))[:, :3]  # 只保留xyz坐标

        sensor_data = SensorData(
            frame=point_cloud.frame,
            timestamp=point_cloud.timestamp,
            data=points
        )
        sensor_queue.put(sensor_data)

    def get_data(self, timeout: float = 1.0) -> Dict[str, np.ndarray]:
        """
        获取传感器数据

        Args:
            timeout: 等待数据的超时时间（秒）

        Returns:
            包含所有传感器数据的字典
        """
        try:
            with self._lock:
                data = {}
                for sensor_name, sensor_queue in self._sensor_queues.items():
                    sensor_data = sensor_queue.get(timeout=timeout)
                    data[sensor_name] = sensor_data.data
                return data
        except queue.Empty:
            raise TimeoutError("Sensor data collection timed out")

    def destroy(self):
        """销毁所有传感器"""
        for sensor in self._sensors.values():
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
        self._sensors.clear()
        self._sensor_queues.clear()


class SensorPreprocessor:
    """传感器数据预处理器"""

    def __init__(self, config):
        """
        初始化预处理器

        Args:
            config: 传感器配置对象
        """
        self.config = config

    def process_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        处理RGB图像

        Args:
            image: 原始图像数据，形状为(H, W, C)

        Returns:
            处理后的图像，形状为(C, H, W)
        """
        # 确保输入是正确的形状
        if image.ndim != 3:
            raise ValueError(f"Expected 3D image array, got shape {image.shape}")

        # 调整大小到模型输入尺寸
        image = cv2.resize(
            image,
            (self.config.rgb_camera['width'],
             self.config.rgb_camera['height'])
        )

        # 标准化到[0, 1]
        image = image.astype(np.float32) / 255.0

        # 转换为PyTorch期望的格式 (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)
        return image

    def process_lidar(self, points: np.ndarray) -> np.ndarray:
        """
        处理LiDAR点云

        Args:
            points: 原始点云数据

        Returns:
            处理后的点云数据，形状为(N, 3)或(N, 4)
        """
        if len(points) == 0:
            # 如果没有点，返回空数组但保持正确的形状
            return np.zeros((0, 4), dtype=np.float32)

        # 随机采样固定数量的点
        n_points = self.config.lidar.get('points_per_frame', 100)

        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
            points = points[indices]
        elif len(points) < n_points:
            # 如果点太少，进行重复采样
            indices = np.random.choice(len(points), n_points - len(points))
            points = np.vstack([points, points[indices]])

        # 标准化坐标
        points[:, :3] = points[:, :3] / self.config.lidar['range']
        return points


