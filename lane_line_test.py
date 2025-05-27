import numpy as np

from WebotsEnvironment import WebotsEnvironment

env = WebotsEnvironment()

print("Finished")

while True:
   
    values = []
        
    for dist_sen in env.dist_sensors:
        values.append(dist_sen.max_value - dist_sen.getValue())
        
    values = np.array(values)
    sum_val = values.sum()
    
    env.robot.step(100)
    