"""av_challenge_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Camera
from vehicle import Car, Driver
import random
import numpy as np
# create the Robot instance.
robot = Driver()
front_camera = robot.getCamera("front_camera")
#rear_camera = robot.getCamera("rear_camera")

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)


front_camera.enable(20)
#rear_camera.enable(30)
lidar = robot.getLidar("Sick LMS 291")
lidar.enable(timestep)
lidar.enablePointCloud()

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step() != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    speed = 50
    # Process sensor data here.
    x  = lidar.getRangeImage()
    # print(robot.getControlMode())
    # print(robot.getThrottle())
    robot.setGear(1)
    
    steer_angle = 0

    robot.setBrakeIntensity(0.0)
    robot.setSteeringAngle(steer_angle)
    robot.setCruisingSpeed(speed)

