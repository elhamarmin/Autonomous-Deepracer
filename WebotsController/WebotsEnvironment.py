import datetime
import math

import cv2
import numpy as np
from controller import Camera, Gyro, Robot, Supervisor
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
        self.lines_count = 0
        self.num_steps_out = 0
        
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
    
    def get_camera_image(self):
        
        camera : Camera = self.camera
        
        pointer = camera.getImage()
        
        image_array = np.frombuffer(pointer, dtype=np.uint8)

        # Reshape the array to the shape (height, width, 4) assuming the image has 4 channels (RGBA)
        img = image_array.reshape((camera.height, camera.width, 4))
        rgb_array = img[:, :, :3]
    
               
        return rgb_array

        
    def get_observations(self):
        

        cam_img = self.get_camera_image()
        left_weight,right_weight,warped,lines = pipeline(cam_img,center_margin=20)

        self.diff_angle = calculate_steering_angle(cam_img.shape[1],left_weight,right_weight)
        self.right_weight = right_weight
        self.left_weight = left_weight 
        self.lines_count = lines

        state_vector = np.array([self.diff_angle,right_weight,left_weight,lines], dtype=np.uint8)
        return state_vector
        
    def reset(self):
        
        self.right_weight = 0
        self.left_weight = 0
        self.diff_angle = 0
        self.prev_diff_angle = 0
        self.lines_count = 0
        self.num_steps_out = 0
        
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

    def get_heading_vector(self):  
        compass = np.array(self.compass.getValues()[0:2], dtype=np.float16)

        return compass
    
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
        previous_position = self.previous_pos
        current_position = np.array(self.gps.getValues()[0:2])
        compass_heading = self.get_heading_vector()

        displacement = current_position - previous_position

        forward_direction = compass_heading / np.linalg.norm(compass_heading)

        forward_direction_3d = np.array([forward_direction[0], forward_direction[1]])
        forward_distance = np.dot(displacement, forward_direction_3d)
        
        return forward_distance
            
            
    def get_reward(self):
        
        self.current_step += 1
        done = False
        
        distance_sensor_weight = -15
        distance_weight = 10
        speed_weight = 1
        angle_weight = 3
        collision_penalty = -100
        
        # dist_sensor = self.get_dist_sensor_value()
        # dist_sensor_reward = dist_sensor * distance_sensor_weight    
    
        distance = self.traveled_distance()
        distance_reward = distance * distance_weight
        
        # imu_value = self.get_imu_value()
        # imu_reward = imu_value * speed_weight
        
        diff = (abs(self.prev_diff_angle) - abs(self.diff_angle))
        angle_reward = 0
        if diff > 0:
            angle_reward = diff * angle_weight
        
        if self.lines_count == 0:
            self.num_steps_out += 1
        
        
        if self.num_steps_out >= 4:
            self.num_steps_out = 0
            angle_reward = -2 * angle_weight
            
            
        collision_reward = 0
        check_collision = self.touch.value
        if check_collision:
            done = True
            collision_reward = collision_penalty
    
        
        reward = angle_reward + collision_reward + distance_reward

        print(f"Reward: {reward:.2f}, dist: {distance_reward:.2f}, angle: {angle_reward:.2f}, done: {done}",end='\r')
        
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
  