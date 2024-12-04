#!/usr/bin/env python3


import sys
import rospy

from new_driver import Driver

from math import atan2, sqrt

import numpy as np


class StudentDriver(Driver):
	'''
	This class implements the logic to move the robot to a specific place in the world.  All of the
	interesting functionality is hidden in the parent class.
	'''
	def __init__(self, threshold=0.1):
		super().__init__('odom')
		# Set the threshold to a reasonable number
		self.robot_width = 0.38
		self._threshold = self.robot_width * 2 # Robot's width NOTE: This needs to be tuned, it's random

	def close_enough_to_waypoint(self, distance, target, lidar):
		'''
		This function is called perioidically if there is a waypoint set.  This is where you should put any code that
		has a smarter stopping criteria then just checking the distance. See get_twist for the parameters; distance
		is the current distance to the target.
		'''
		# Default behavior.
		if distance < self._threshold:
			return True
		return False
	
	def get_twist(self, target, lidar):
		'''
		This function is called whenever there a current target is set and there is a lidar data
		available.  This is where you should put your code for moving the robot.  The target point
		is in the robot's coordinate frame.  The x-axis is positive-forwards, and the y-axis is
		positive to the left.

		The example sets constant velocities, which is clearly the wrong thing to do.  Replace this
		code with something that moves the robot more intelligently.

		Parameters:
			target:		The current target point, in the coordinate frame of the robot (base_link) as
						an (x, y) tuple.
			lidar:		A LaserScan containing the new lidar data.

		Returns:
			A Twist message, containing the commanded robot velocities.
		'''
		angle = atan2(target[1], target[0])
		distance = sqrt(target[0] ** 2 + target[1] **2)
		rospy.loginfo(f'Distance: {distance:.2f}, angle: {angle:.2f}')

		w = 0.38
		command = Driver.zero_twist()
		thetas = np.linspace(lidar.angle_min, lidar.angle_max, len(lidar.ranges))
		ranges = np.array(lidar.ranges)

		# Get the indexes of scans in front of the robot where the obstacle is 2 meters away or closer
		in_front_idx = np.where((ranges * np.abs(np.sin(thetas)) <= w/2) & (ranges < 1))[0]

		# If nothing is in front of us within 2 meters
		if len(in_front_idx) == 0:
			#  Step 1) Calculate the angle the robot has to turn to in order to point at the target
			target_angle = np.arctan2(target[1], target[0])
			target_distance = np.linalg.norm(np.array(target))

			#  Step 2) Set your speed based on how far away you are from the target, as before
			command.linear.x = target_distance / 2
			command.angular.z = target_angle * 0.75
		else:
			#  Step 3) Add code that veers left (or right) to avoid an obstacle in front of it
			left = 0
			right = 0

			# Sum space to the left and right of the robot
			for r, theta in zip(ranges, thetas):
				if theta < 0:
					right += r
				else:
					left += r
			
			# Turn in the direction with more space
			if right > left:
				command.angular.z = -0.2
			else:
				command.angular.z = 0.2

			# Don't move forward if there is something within 1 meter, move slowly if not
			# NOTE: THIS WAS CHANGED, BEWARE; ALSO USED TO BE * 0.1
			command.linear.x = 0 if np.min(ranges[in_front_idx]) < 1 else np.min(ranges[in_front_idx]) #* 0.1
			# command.linear.x = 0.2

		return command


if __name__ == '__main__':
	rospy.init_node('student_driver', argv=sys.argv)

	driver = StudentDriver()

	rospy.spin()
