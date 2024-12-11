#!/usr/bin/env python3


import sys
import rospy

from new_driver import Driver

from math import atan2, sqrt, ceil, tanh, pi
import numpy as np


class StudentDriver(Driver):
	'''
	This class implements the logic to move the robot to a specific place in the world.  All of the
	interesting functionality is hidden in the parent class.
	'''
	def __init__(self, threshold=0.1):
		super().__init__('odom')
		# Set the threshold to a reasonable number
		self._robot_width = 0.38
		self._threshold = self._robot_width * 2

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
		w_epsilon = 0.1
		w = 0.38 # Robot's width
		l = 0.44
		command = Driver.zero_twist() 
		thetas = np.linspace(lidar.angle_min, lidar.angle_max, len(lidar.ranges))
		ranges = np.array(lidar.ranges)

		obstacle_threshold = 0.3 + (l / 2)
		obstacles_in_front_idx = np.where((ranges * np.abs(np.sin(thetas)) <= w/2) & (ranges < obstacle_threshold))[0]

		if (len(obstacles_in_front_idx) == 0):
			target_angle = atan2(target[1], target[0])
			target_distance = np.linalg.norm(np.array(target))

			if abs(target_angle) > pi / 2:
				self.rotate_180()

			command.linear.x = target_distance / 2
			command.angular.z = target_angle * 0.75
			return command
		else:
			obstacle_distance = np.min(ranges[obstacles_in_front_idx]) - l / 2

			angle_of_concern = 2 * abs(np.arctan(w / 2 / obstacle_threshold))
			angle_per_scan = ((lidar.angle_max - lidar.angle_min) / len(lidar.ranges))
			num_scans_of_concern = ceil(angle_of_concern / angle_per_scan)

			cones = np.lib.stride_tricks.sliding_window_view(ranges, num_scans_of_concern)
			safe_cones_idx = np.nonzero(np.all(cones > obstacle_threshold, axis=1))[0]

			if len(safe_cones_idx) == 0:
				self.rotate_180()
				return command

			nearest_safe_cone_idx = safe_cones_idx[np.argmin(np.abs(safe_cones_idx - (len(cones) / 2)))]

			half_window = num_scans_of_concern / 2
			if half_window % 1 == 0:
				lower_idx = nearest_safe_cone_idx + int(half_window)
				safe_direction = (thetas[lower_idx] + thetas[lower_idx + 1]) / 2
			else:
				safe_direction = thetas[nearest_safe_cone_idx + int(half_window)]

			rospy.loginfo(f"SAFE DIRECTION IS: {safe_direction} OBSTACLE DISTANCE: {obstacle_distance}")
			command.angular.z = 4 * tanh(1 * safe_direction * (1 / obstacle_distance)) + 1 if safe_direction > 0 else -1
			command.linear.x = 0.5 * tanh(1 * (1 / obstacle_distance)) if obstacle_distance > 0.25 else 0

		return command


if __name__ == '__main__':
	rospy.init_node('student_driver', argv=sys.argv)

	driver = StudentDriver()

	rospy.spin()
