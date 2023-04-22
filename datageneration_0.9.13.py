import argparse
import logging
import os
import os.path as osp
from queue import Queue
import random
import time

import carla
from carla import WeatherParameters as wp
import numpy as np
from tqdm import tqdm

from constants import *


""" DATA SAVE PATHS """
SEQ_FOLDER = "output/sequences/{}"
POSE_FOLDER = "output/poses/"
FOLDERS = ['image_2', 'label_2', 'velodyne']

CALIBRATION_PATH = 'calib.txt'
LIDAR_PATH = 'velodyne/{:06}.bin'
IMAGE_PATH = 'image_2/{:06}.png'
LABEL_PATH = 'label_2/{:06}.txt'
TIME_PATH = 'times.txt'
POSE_PATH = '{:06}.txt'

""" Carla constants """
CITIES = ['Town01', 'Town04', 'Town05']
WEATHERS = [wp.ClearNight, wp.ClearNoon, wp.ClearSunset,
            wp.CloudyNight, wp.CloudyNoon, wp.CloudySunset,
            wp.WetNight, wp.WetNoon, wp.WetSunset,
            wp.WetCloudyNight, wp.WetCloudyNoon, wp.WetCloudySunset,
            wp.MidRainyNight, wp.MidRainyNoon, wp.MidRainSunset,
            wp.HardRainNight, wp.HardRainNoon, wp.HardRainSunset,
            wp.SoftRainNight, wp.SoftRainNoon, wp.SoftRainSunset]


class CarlaGame(object):

    def __init__(self, client, seq_id):
        self.client = client
        self.frame_id = -1
        self._rgb_timestamps = Queue()  # use queue to ensure synchronization
        self._lidar_timestamps = Queue()
        self._depth_queue = Queue()
        self._create_dirs(seq_id)
        self._set_world()
        self._pbar = tqdm(total=NUM_RECORDINGS_BEFORE_RESET)

    def _create_dirs(self, seq_id):
        seq_root = SEQ_FOLDER.format(seq_id)
        os.makedirs(POSE_FOLDER, exist_ok=True)
        for folder in FOLDERS:
            os.makedirs(os.path.join(seq_root, folder), exist_ok=True)

        self._lidar_path = osp.join(seq_root, LIDAR_PATH)
        self._label_path = osp.join(seq_root, LABEL_PATH)
        self._image_path = osp.join(seq_root, IMAGE_PATH)
        self._calib_path = osp.join(seq_root, CALIBRATION_PATH)
        self._time_path = osp.join(seq_root, TIME_PATH)
        self._pose_path = osp.join(POSE_FOLDER, POSE_PATH)
    
    def _set_world(self):
        self._city = random.choice(CITIES)
        logging.info(f'Loading world: {self._city} ...')
        self._world = self.client.load_world(self._city)
        self._original_settings = self._world.get_settings()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.1
        self._world.apply_settings(settings)

        # set weather
        weather = random.choice(WEATHERS)
        self._world.set_weather(weather)

        # set all traffic light to green
        for light in self._world.get_actors().filter('traffic.traffic_light'):
            light.set_state(carla.TrafficLightState.Green)
            light.set_green_time(1000.0)
            light.set_red_time(1.0)
            light.set_yellow_time(1.0)

        # set traffic
        bplib = self._world.get_blueprint_library()
        spawn_points = random.choices(self._world.get_map().get_spawn_points(), k=NUM_VEHICLES)
        ego_vehicle_bp = bplib.find('vehicle.micro.microlino')
        ego_vehicle_bp.set_attribute('role_name', 'hero')
        vehicle_bps = random.choices(list(bplib.filter('vehicle')), k=NUM_VEHICLES-1)
        vehicle_bps = [ego_vehicle_bp] + vehicle_bps
        self._vehicles = []
        for vehicle_bp, transform in zip(vehicle_bps, spawn_points):
            vehicle = self._world.try_spawn_actor(vehicle_bp, transform)
            if vehicle:
                vehicle.set_autopilot(True)
                self._vehicles.append(vehicle)
        self._ego_vehicle = self._vehicles[0]
        logging.info(f'{len(self._vehicles)} vehicles spawned')

        walker_bps = random.choices(list(bplib.filter('walker')), k=NUM_PEDESTRIANS)
        self._walkers = []
        for walker_bp in walker_bps:
            location = self._world.get_random_location_from_navigation()
            transform = carla.Transform()
            transform.location = location
            walker = self._world.try_spawn_actor(walker_bp, transform)
            if walker:
                self._walkers.append(walker)
        logging.info(f'{len(self._walkers)} pedestrians spawned')

        # set walker controller
        controller_bp = bplib.find('controller.ai.walker')
        for walker in self._walkers:
            controller = self._world.spawn_actor(controller_bp, walker.get_transform(), walker)
            controller.start()
            controller.go_to_location(self._world.get_random_location_from_navigation())

        # set sensors
        camera_bp = bplib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        self._camera_extrin = carla.Transform(carla.Location(z=CAMERA_HEIGHT_POS))
        self._rgb_camera = self._world.spawn_actor(camera_bp, self._camera_extrin, attach_to=self._ego_vehicle)
        self._rgb_camera.listen(self._on_rgb_image)

        lidar_bp = bplib.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', str(MAX_RENDER_DEPTH_IN_METERS))
        lidar_bp.set_attribute('rotation_frequency', str(10))
        lidar_bp.set_attribute('channels', str(40))
        lidar_bp.set_attribute('points_per_second', str(720000))
        lidar_bp.set_attribute('upper_fov', str(7))
        lidar_bp.set_attribute('lower_fov', str(-16))
        lidar_bp.set_attribute('rotation_frequency', str(1.0 / settings.fixed_delta_seconds))
        self._lidar_extrin = carla.Transform(carla.Location(z=LIDAR_HEIGHT_POS))
        self._lidar = self._world.spawn_actor(lidar_bp, self._lidar_extrin, attach_to=self._ego_vehicle)
        self._lidar.listen(self._on_lidar_data)

        depth_camera_bp = bplib.find('sensor.camera.depth')
        depth_camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        depth_camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        depth_camera_bp.set_attribute('fov', str(90))
        # self._depth_camera = self._world.spawn_actor(depth_camera_bp, self._camera_extrin, attach_to=self._ego_vehicle)
        # self._depth_camera.listen(self._on_depth_image)
    
    def reset_world(self):
        self._world.apply_settings(self._original_settings)

    def _on_rgb_image(self, image):
        logging.debug(f'Image frame: {image.frame}, timestamp: {image.timestamp}')
        image.save_to_disk(self._image_path.format(self.frame_id), carla.ColorConverter.Raw)
        self._rgb_timestamps.put(image.timestamp)

    def _on_lidar_data(self, lidar_data):
        logging.debug(f'Lidar frame: {lidar_data.frame}, timestamp: {lidar_data.timestamp}')
        pcd_size = len(lidar_data)
        pcd = np.copy(np.frombuffer(lidar_data.raw_data, dtype=np.float32))
        pcd = np.reshape(pcd, (pcd_size, 4))
        pcd[:, 1] *= -1  # refer to save_lidar_data in dataexport.py
        pcd.tofile(self._lidar_path.format(self.frame_id))
        self._lidar_timestamps.put(lidar_data.timestamp)
    
    def _on_depth_image(self, image):
        self._depth_queue.put(image)
    
    def _save_calibration(self, camera2ego, lidar2ego):
        # TODO
        pass

    def execute(self):
        try:
            for _ in range(NUM_RECORDINGS_BEFORE_RESET):
                self.tick()
        finally:
            # self.destroy_actors()
            pass

    def tick(self):
        self.frame_id += 1
        self._world.tick()
        if self.frame_id == NUM_RECORDINGS_BEFORE_RESET:
            return False
        # TODO: save pose, time, calibration, label
        ego_pose = self._ego_vehicle.get_transform()

        camera_timestamp = self._rgb_timestamps.get(block=True, timeout=3.0)
        lidar_timestamp = self._lidar_timestamps.get(block=True, timeout=3.0)
        assert camera_timestamp == lidar_timestamp
        self._pbar.update(1)
        return True

    def destroy_actors(self):
        actors = self._vehicles + self._walkers + [self._rgb_camera, self._lidar] # , self._depth_camera]
        for actor in actors:
            actor.destroy()


def parse_args():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='logging.info debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--static_world',
        action='store_true',
        help='set the world to be static (except ego vehicle)')
    argparser.add_argument(
        '--start_seq_id',
        default=0,
        type=int,
        help='start sequence id (default: 0)')
    args = argparser.parse_args()
    return args


def main():
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    logging.info(__doc__)

    seq_id = args.start_seq_id
    while True:
        if seq_id == 100:
            break
        try:
        # if True:
            client = carla.Client(str(args.host), int(args.port))
            game = CarlaGame(client, '{:02}'.format(seq_id))
            logging.info(f'Start generating data of sequence {seq_id}')
            game.execute()
            game.reset_world()
            seq_id += 1
        except Exception as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.info('\nCancelled by user. Bye!')