import math
import random

import cv2
import numpy as np

from MyImageProccess.helper_function import (perspective_transform,
                                             region_of_interest)


def calculate_steering_angle(image_width, left_weight, right_weight):
    final_weight = right_weight - left_weight

    diff = image_width / 2
    angle = final_weight / diff
    return angle


def pipeline(image, center_margin = 50):
    """
    An image processing pipeline which will output
    an image with the lane lines annotated.
    """
    
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height),
        (width * 0.25, height * 0.60),
        (width * 0.75, height * 0.60),
        (width, height),
    ]

    cropped_image = region_of_interest(
        image,
        np.array(
            [region_of_interest_vertices],
            np.int32
        ),
    )

    upper_yellow = np.array([215, 220, 237])
    lower_yellow = np.array([80, 80, 208])

    mask = cv2.inRange(cropped_image, lower_yellow, upper_yellow)
    masked = cv2.bitwise_and(image, image , mask=mask)

    gray_image = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)

    warped, M, M_inv = perspective_transform(gray_image)
    cannyed_image = cv2.Canny(warped, 850, 250, apertureSize=3)

    lines = cv2.HoughLinesP(cannyed_image, rho=2.5, theta=np.pi/180, threshold=15, minLineLength=20, maxLineGap=15)
        
    left_points = []
    right_points = []

    center = cannyed_image.shape[1] // 2
    

    left_weight = 0
    right_weight = 0

    def calc_weight(x):
        return abs(x - center)
    
    if lines is None:
        return left_weight,right_weight,left_points,right_points,cannyed_image,warped
    
    for line in lines:
        for x1, y1, x2, y2 in line:

            if abs(x1 - x2) > 5 and abs(x1 - x2) < 10:
                continue  # Skip vertical lines to avoid division by zero
            
            if x1 < center - center_margin :  # Left line
                left_points.append((x1, y1))
                left_weight += calc_weight(x1)
            if x2 < center - center_margin :  # Left line
                left_points.append((x2, y2))
                left_weight += calc_weight(x2)
            
            if x1 > center + center_margin :  # Left line
                right_points.append((x1, y1))
                right_weight += calc_weight(x1)
                
            if x2 > center + center_margin :  # Left line
                right_points.append((x2, y2))
                right_weight += calc_weight(x2)
                
    if len(right_points) > 0:
        right_weight = right_weight / len(right_points)

    if len(left_points) > 0:
        left_weight = left_weight / len(left_points)
         
    return left_weight,right_weight,left_points,right_points,cannyed_image,warped