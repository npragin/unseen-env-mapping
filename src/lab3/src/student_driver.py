#!/usr/bin/env python3


import sys
import rospy

from new_driver import Driver

from math import atan2, ceil, tanh, pi
import numpy as np


class StudentDriver(Driver):
	'''
	This class implements the logic to move the robot to a specific place in the world.  All of the
	interesting ROS-specific functionality is hidden in the parent class.
	'''
	def __init__(self):
		super().__init__('odom')

		self._robot_width = 0.38
		self._robot_length = 0.44
		self._robot_diagonal_length = np.linalg.norm((self._robot_length, self._robot_width))

		self._lidar_offset = self._robot_diagonal_length / 2

	def close_enough_to_waypoint(self, distance, target, lidar):
		"""
		Determines if the robot is close enough to the waypoint to mark it as complete.

		This function is called periodically if there is a waypoint set. It checks if the 
		waypoint is beneath the robot by using the robot's dimensions.

		Parameters:
			distance (float): Distance to the target.
			target (tuple): A tuple (x, y) representing the distance to the target.
			lidar (LaserScan): A LaserScan containing the new lidar data.

		Returns:
			bool: True if the robot is close enough to the waypoint to consider it
				  reached, False otherwise.
		"""
		# NOTE: The height and and width should be halved to determine if the goal is beneath the robot
		if abs(target[0]) <= self._robot_length and abs(target[1]) <= self._robot_width:
			return True
		return False

	def get_twist(self, target, lidar):
		'''
		Generates a twist command to drive to the current target point with obstacle avoidance.

		This function is called whenever there is a current target set and lidar data is available.
		This is where the code for moving the robot is implemented. The target point is in the robot's
		coordinate frame, with the x-axis positive-forwards and the y-axis positive to the left.
		Obstacles within the obstacle threshold and in front of the robot are avoided.

		Parameters:
			target (tuple): The current target point, in the coordinate frame of the robot (base_link) as
							an (x, y) tuple.
			lidar (LidarScan): A LaserScan containing the new lidar data.

		Returns:
			A Twist message, containing the commanded robot velocities.
		'''
		# Initialize twist command with zero values to be populated below
		command = Driver.zero_twist()

		# We will use the diagonal length of the robot to determine if there is something
		# in the robot's path. We use the diagonal length instead of the width to ensure
		# the robot can always rotate without collision.
		d = self._robot_diagonal_length

		# Extracting relevant data from the lidar into numpy arrays
		thetas = np.linspace(lidar.angle_min, lidar.angle_max, len(lidar.ranges))
		ranges = np.array(lidar.ranges)

		# The obstacle threshold is the distance an obstacle must be from the robot to
		# consider an obstacle in the robot's path. We use half the diagonal length to ensure
		# the robot can rotate without collision. We add half the robot's diagonal length to
		# account for the fact that the lidar is in the center of the robot. We then use
		# this distance with some trigonometry to gather the indices of lidar readings
		# that correspond to scans in front of the robot and have a reading less than the
		# obstacle threshold. See issue #55 for more info.
		obstacle_threshold = d / 2 + self._lidar_offset
		obstacles_in_front_idx = np.where((ranges * np.abs(np.sin(thetas)) <= d/2) & (ranges < obstacle_threshold))[0]

		# The angle from the robot to the target point
		target_angle = atan2(target[1], target[0])

		# If we have no obstacles in front of the robot, we can move towards the target
		if (len(obstacles_in_front_idx) == 0):
			target_distance = np.linalg.norm(np.array(target))

			# If the target is behind the robot, rotate to face the target
			if abs(target_angle) > pi / 2:
				if self._rotate_count == 0:
					rospy.loginfo('Rotating to face goal')
					self._rotate_count += self._get_rotate_count(target_angle)
					return command

			# Set the linear and angular velocity to a linear scale of the distance and
			# angle to the target
			command.linear.x = target_distance
			command.angular.z = target_angle * 0.75

		# If there are obstacles in front of the robot, we need to avoid them
		else:
			# Get the obstacle distance by finding the minimum reading from the lidar of
			# the scans in front of the robot. Subtract half the lidar offset from the
			# minimum distance to an obstacle to account for the fact that the lidar is
			# in the center of the robot, so the obstacle is closer than the lidar reading indicates.
			obstacle_distance = np.min(ranges[obstacles_in_front_idx]) - self._lidar_offset

			# Given the obstacle distance and the robot's diagonal length, we can calculate
			# the "angle of concern," the angle of the cone that must be free of obstacles
			# for the robot to continue toward the goal. Using this angle, we can determine
			# the number of adjacent scans that must have a reading greater than the
			# obstacle threshold to consider that direction safe.
			angle_of_concern = 2 * abs(np.arctan(obstacle_threshold / 2 / obstacle_distance))
			angle_per_scan = ((lidar.angle_max - lidar.angle_min) / len(lidar.ranges))
			num_scans_of_concern = ceil(angle_of_concern / angle_per_scan)

			# Using the number of scans of concern, we use a sliding window to find all
			# the cones that are free of obstacles.
			cones = np.lib.stride_tricks.sliding_window_view(ranges, num_scans_of_concern)
			safe_cones_idx = np.nonzero(np.all(cones > obstacle_threshold, axis=1))[0]

			# If there are no safe cones, rotate to face the goal. We can face the goal
			# instead of rotating 180 degrees (which you might want to do to find free
			# space given the lidar's field of view is 180 degrees) because the waypoints
			# are generated in the configuration space of the robot and such that you
			# can travel in a straight line between them. Thus, waypoints will be reachable
			# without needing to rotate in a way that would take the goal point out of the
			# lidar's field of view
			if len(safe_cones_idx) == 0:
				if self._rotate_count == 0:
					rospy.loginfo('Rotating to face free space')
					self._rotate_count += self._get_rotate_count(target_angle)
				return command

			# If there is a safe cone, we want to move in the direction of the cone that
			# requires the least rotation, we find its index
			nearest_safe_cone_idx = safe_cones_idx[np.argmin(np.abs(safe_cones_idx - (len(cones) / 2)))]

			# Using the index of the nearest safe cone, we calculate the direction the
			# robot should face
			half_window = num_scans_of_concern / 2
			if half_window % 1 == 0:
				lower_idx = nearest_safe_cone_idx + int(half_window)
				safe_direction = (thetas[lower_idx] + thetas[lower_idx + 1]) / 2
			else:
				safe_direction = thetas[nearest_safe_cone_idx + int(half_window)]

			# Use a carefully tuned tanh function to turn faster as obstacles get closer
			# and drive slower as obstacles get closer.
			command.angular.z = 4 * tanh(1 * safe_direction * (1 / obstacle_distance)) * (1 if safe_direction > 0 else -1)
			command.linear.x = 0.5 * tanh(1 * obstacle_distance) if obstacle_distance > 0.25 else 0

		return command


if __name__ == '__main__':
	rospy.init_node('student_driver', argv=sys.argv)

	driver = StudentDriver()

	rospy.spin()
