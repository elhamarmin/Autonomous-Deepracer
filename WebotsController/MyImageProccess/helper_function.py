import math

import cv2
import numpy as np


def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def perspective_transform(image,roi=None):
    height, width = image.shape[:2]
    
    if roi is not None:
        src = np.float32([(0, height),
                            ((width * 0.25), height * 0.55),
                            ((width * 0.75), height * 0.55),
                            (width, height),])    # top-right
    else:
        src = roi
    
    dst = np.float32([(0, height),
                        (100, 100),
                        (width - 100, 100),
                        (width, height)])
    
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped, M, M_inv