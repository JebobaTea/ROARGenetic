"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""
import math

import roar_py_interface
import numpy as np
import os
import random
from collections import deque
from typing import List, Tuple, Dict, Optional
from GeneralizedFeedforwardModel import GeneralizedFeedforwardModel


def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
        model: GeneralizedFeedforwardModel = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_pid_controller = LatPIDController(config=self.get_lateral_pid_config())
        self.model = model
        self.coeff = 1
        self.steps = 0

    def get_lateral_pid_config(self):
        conf = {
            "60": {
                "Kp": 0.7,
                "Kd": 0.05,
                "Ki": 0.05
            },
            "70": {
                "Kp": 0.7,
                "Kd": 0.07,
                "Ki": 0.07
            },
            "80": {
                "Kp": 0.65,
                "Kd": 0.08,
                "Ki": 0.08
            },
            "90": {
                "Kp": 0.57,
                "Kd": 0.13,
                "Ki": 0.09
            },
            "100": {
                "Kp": 0.5,
                "Kd": 0.15,
                "Ki": 0.1
            },
            "120": {
                "Kp": 0.4,
                "Kd": 0.15,
                "Ki": 0.1
            },
            "130": {
                "Kp": 0.35,
                "Kd": 0.15,
                "Ki": 0.09
            },
            "140": {
                "Kp": 0.3,
                "Kd": 0.15,
                "Ki": 0.09
            },
            "160": {
                "Kp": 0.25,
                "Kd": 0.3,
                "Ki": 0.06
            },
            "180": {
                "Kp": 0.15,
                "Kd": 0.25,
                "Ki": 0.05
            },
            "200": {
                "Kp": 0.01,
                "Kd": 0.4,
                "Ki": 0.04
            },
            "230": {
                "Kp": 0.01,
                "Kd": 0.4,
                "Ki": 0.05
            },
            "300": {
                "Kp": 0.02,
                "Kd": 0.3,
                "Ki": 0.017
            }
        }
        return conf


    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                # waypoints9 best
                np.load(f"{os.path.dirname(__file__)}\\waypointsPrimary.npz") # Lifted from DerekChen1
            )
        )

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )


    async def step(
        self
    ) -> None:
        # Obtain vehicle data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_speed = np.linalg.norm(vehicle_velocity) * 3.6

        # Get nearest waypoint and determine target waypoints
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location, self.current_waypoint_idx, self.maneuverable_waypoints
        )
        waypoint_to_follow = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints,
                                                                            self.current_waypoint_idx, 3)
        far_waypoint = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints,
                                                                      self.current_waypoint_idx, 25)
        near_waypoint = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints,
                                                                      self.current_waypoint_idx, 10)
        lookahead = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints,
                                                                   self.current_waypoint_idx, int(vehicle_speed / self.coeff))

        # Steering control
        steer_control = self.lat_pid_controller.run_in_series(
            vehicle_location, vehicle_rotation, vehicle_speed, waypoint_to_follow
        )
        far_error = self.lat_pid_controller.find_waypoint_error(
            vehicle_location, vehicle_rotation, vehicle_speed, far_waypoint
        )
        near_error = self.lat_pid_controller.find_waypoint_error(
            vehicle_location, vehicle_rotation, vehicle_speed, near_waypoint
        )
        dynamic_error = self.lat_pid_controller.find_waypoint_error(vehicle_location, vehicle_rotation, vehicle_speed, lookahead)

        model_input = np.array([
            [near_error, far_error, dynamic_error, vehicle_speed/300, steer_control]
        ])
        model_output = self.model.feed_forward(model_input)

        # Dynamic throttle and brake control
        throttle = abs(model_output.item(0))
        self.coeff = model_output.item(1)*10
        brake = abs(model_output.item(2))

        if (vehicle_speed < 10 and brake < throttle):
            # help accelerate from a dead stop, but only allow vehicle to live if aggressive enough
            brake = 0

        # Logging for debugging
        #os.system("cls")
        self.steps += 1
        if self.steps % 500 == 0 or self.steps == 100:
            print(f"Brake: {brake:.2f}, Throttle: {throttle:.2f}, Speed: {vehicle_speed:.2f}, Distance: {self.current_waypoint_idx}")
        # Apply vehicle controls
        control = {
            "throttle": np.clip(throttle, 0.0, 1.0),
            "steer": np.clip(steer_control, -1.0, 1.0),
            "brake": np.clip(brake, 0.0, 1.0),
            "hand_brake": 0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control

class LatPIDController():
    def __init__(self, config: dict, dt: float = 0.05):
        self.config = config
        self.steering_boundary = (-1.0, 1.0)
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    def run_in_series(self, vehicle_location, vehicle_rotation, current_speed, next_waypoint) -> float:
        v_begin = vehicle_location
        direction_vector = np.array([
            np.cos(normalize_rad(vehicle_rotation[2])),
            np.sin(normalize_rad(vehicle_rotation[2])),
            0])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), (v_end[1] - v_begin[1]), 0])

        w_vec = np.array(
            [
                next_waypoint.location[0] - v_begin[0],
                next_waypoint.location[1] - v_begin[1],
                0,
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1))
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[2] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = self.find_k_values(current_speed=current_speed, config=self.config)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )

        return lat_control

    def find_k_values(self, current_speed: float, config: dict) -> np.array:
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])

    def find_waypoint_error(self, vehicle_location, vehicle_rotation, current_speed, waypoint) -> float:
        v_begin = vehicle_location
        direction_vector = np.array([
            np.cos(normalize_rad(vehicle_rotation[2])),
            np.sin(normalize_rad(vehicle_rotation[2])),
            0])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), (v_end[1] - v_begin[1]), 0])

        w_vec = np.array(
            [
                waypoint.location[0] - v_begin[0],
                waypoint.location[1] - v_begin[1],
                0,
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1))

        return error

    def get_waypoint_at_offset(self, maneuverable_waypoints, current_index, offset):
        return maneuverable_waypoints[(current_index + offset) % len(maneuverable_waypoints)]
