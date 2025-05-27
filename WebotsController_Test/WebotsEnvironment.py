import math

import cv2
import numpy as np
from controller import Gyro, Robot, Supervisor
from matplotlib import pyplot as plt

from Interfaces.BaseEnvironment import BaseEnvironment
from MyImageProccess.lane_lines import calculate_steering_angle, pipeline


class WebotsEnvironment(Supervisor,BaseEnvironment):
    """The robot's environment in Webots."""

    def __init__(self, writer=None):     
        self.robot = Robot()
        
        super().__init__()
        
        self.hit_count = 0
        self.reach_goal_count = 0
        
        self.current_step = 0
        self.writer = writer
        # General environment parameters
        self.max_speed = 20  # Maximum Angular speed in rad/s
        self.max_rotation = 4  # Maximum Angular speed in rad/s
        
        self.destination_coordinate = np.array([-3.13, 0])  # Target (Goal) position
        self.reach_threshold = 0.1  # Distance threshold for considering the destination reached.
        self.safe_distance = 0.35
        self.floor_size = np.linalg.norm([8, 8])

        # Activate Devices
        self.bl_motors = []
        
        self.bl_motors.append(self.robot.getDevice('left_front_wheel_motor'))
        self.bl_motors.append(self.robot.getDevice('right_front_wheel_motor'))
        self.bl_motors.append(self.robot.getDevice('left_rear_wheel_motor'))
        self.bl_motors.append(self.robot.getDevice('right_rear_wheel_motor'))

        for motor in self.bl_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        self.servos = []
        
        self.servos.append(self.robot.getDevice('left_front_wheel_steering_motor'))
        self.servos.append(self.robot.getDevice('right_front_wheel_steering_motor'))
        
        self.previous_pos = None
        
        # GPS Sensor
        sampling_period = 1  # in ms
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(sampling_period)
        
        
        self.imu = self.robot.getDevice("accelerometer")
        self.imu.enable(sampling_period)
        
        
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(sampling_period)

        # Touch Sensor
        self.touch = self.robot.getDevice("touch sensor")
        self.touch.enable(sampling_period)
        
        self.compass = self.robot.getDevice("compass")
        self.compass.enable(sampling_period)
        
        self.lidar = self.robot.getDevice("lidar")
        self.lidar.enable(sampling_period)
        
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(sampling_period)
        
        self.dist_sensors = []
        
        self.dist_sensors.append(self.robot.getDevice("dist_sen_right"))
        self.dist_sensors.append(self.robot.getDevice("dist_sen_left"))
        
        for dist_sen in self.dist_sensors:
            dist_sen.enable(sampling_period)

        self.right_weight = 0
        self.left_weight = 0
        self.diff_angle = 0
        self.prev_diff_angle = 0
        
        # Reset
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        self.robot.step(200)  # take some dummy steps in environment for initialization

        
    def reset_rewards(self):
        self.reach_goal_count = 0
        self.hit_count = 0
        
    def set_log_writer(self , writer):
        self.writer = writer
        
        
    def normalizer(self, value, min_value, max_value):
        normalized_value = (value - min_value) / (max_value - min_value)
        return normalized_value
    
    def get_lidar_data(self):
        
        lidar = np.array(self.lidar.getRangeImage())
        lidar = self.normalizer(lidar , 0 , 1.5)
        lidar = np.clip(lidar , 0 , 1)
        
        return lidar
    
    def get_camera_image(self):
        
        camera = self.camera
        img = np.array(camera.getImageArray(), dtype=np.uint8)
        cv2_img = cv2.cvtColor(img.reshape((720,1280,3)) , cv2.COLOR_BGR2RGB)
               
        return cv2_img

        
    def get_lidar_score(self):
        
        lidar_data = self.get_lidar_data()
        
        min_distance = np.min(lidar_data)
        if min_distance < self.safe_distance:
            return -5 * (self.safe_distance - min_distance)  # Penalty for being too close to walls
        else:
            return 1.6  # Reward for maintaining a safe distance


        
    def get_observations(self):
        
        cam_img = self.get_camera_image()
        left_weight,right_weight,left_points,right_points,cannyed_image,warped = pipeline(cam_img)
        
        self.diff_angle = calculate_steering_angle(cannyed_image.shape[1],left_weight,right_weight)
        self.right_weight = right_weight
        self.left_weight = left_weight
        
        # lidar_data = self.get_lidar_data()
        
        # gps_value = self.gps.getValues()[0:2]
        # norm_gps_value = gps_value / np.linalg.norm(gps_value)
        # vector_goal = self.get_vector_to_goal()
        
        # compass_data = np.array(self.compass.getValues()[0:2])

        state_vector = np.concatenate([warped.flatten()], dtype=np.uint8)

        return state_vector
        
    def reset(self):
        if self.writer != None:
            self.writer.log({'hit_count' : self.hit_count , 'goal_count' : self.reach_goal_count})
        
        self.right_weight = 0
        self.left_weight = 0
        self.diff_angle = 0
        self.prev_diff_angle = 0
        
        
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        
        return self.get_observations()

    def step(self, action, max_steps):
        self.previous_pos = np.array(self.gps.getValues()[0:2])
        
        self.apply_action(action)
        state = self.get_observations()  # New state
        
        step_reward, done = self.get_reward()

        self.prev_diff_angle = self.diff_angle
        
        truncated = False
        # Time-based termination condition
        if (int(self.getTime()) + 1) % max_steps == 0:
            truncated = True

        return state, step_reward, done, truncated


    def get_imu_value(self):
        values = -np.array(self.imu.getValues()[:2])

        speed = np.linalg.norm(values) * np.sign(values[0])
        return speed
    
    def get_gyro_value(self):
        values = self.gyro.getValues()[2]
        
        return values[2]
    
    def get_dist_sensor_value(self):
        values = []
        max_val = self.dist_sensors[0].max_value
        
        for dist_sen in self.dist_sensors:
            values.append(dist_sen.max_value - dist_sen.getValue())
            
        values = np.array(values)
        sum_val = values.sum() / max_val
        
        return sum_val
    
    
    def traveled_distance(self):
        distance = 0
        
        current_pos = np.array(self.gps.getValues()[0:2])
        if self.previous_pos is not None:
            dist_vec = current_pos - self.previous_pos
            
            values = -np.array(dist_vec)
            distance = np.linalg.norm(values) * np.sign(values[0])

        
        return distance
            
            
    def get_reward(self):
        
        self.current_step += 1
        done = False
        
        distance_sensor_weight = -15
        distance_weight = 12
        speed_weight = 3
        angle_weight = 20
        collision_penalty = -100
        
        # dist_sensor = self.get_dist_sensor_value()
        # dist_sensor_reward = dist_sensor * distance_sensor_weight    
    
        # distance = self.traveled_distance()
        # distance_reward = distance * distance_weight
        
        # imu_value = self.get_imu_value()
        # imu_reward = imu_value * speed_weight
        
        angle_reward = (abs(self.prev_diff_angle) - abs(self.diff_angle)) * angle_weight
        
        if self.right_weight == 0 and self.left_weight == 0:
            angle_reward = -5 * angle_weight
        
        collision_reward = 0
        
        check_collision = self.touch.value
        if check_collision:
            done = True
            collision_reward = collision_penalty
    
            
        reward = angle_reward + collision_reward

        return reward, done


    def apply_action(self, action):
        
        for motor in self.bl_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(self.max_speed * action[0])
            
        for motor in self.servos:
            motor.setPosition(action[1])

        self.robot.step(250)
        
        for motor in self.bl_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0)
  