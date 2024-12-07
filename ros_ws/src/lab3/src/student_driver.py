#!/usr/bin/env python3


import sys
import rospy

from new_driver import Driver

from math import atan2, sqrt


class StudentDriver(Driver):
	'''
	This class implements the logic to move the robot to a specific place in the world.  All of the
	interesting functionality is hidden in the parent class.
	'''
	def __init__(self, threshold=0.1):
		super().__init__('odom')
		# Set the threshold to a reasonable number
		self._threshold = threshold

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
		command = Driver.zero_twist()

		# TODO:
		#  Step 1) Calculate the angle the robot has to turn to in order to point at the target
		#  Step 2) Set your speed based on how far away you are from the target, as before
		#  Step 3) Add code that veers left (or right) to avoid an obstacle in front of it
		# This sets the move forward speed (as before)

		target_angle = atan2(target[1], target[0])
		print(f"target: {target[0]}, {target[1]}")

		angle_min = lidar.angle_min
		angle_max = lidar.angle_max
		num_readings = len(lidar.ranges)
		thetas = np.linspace(angle_min, angle_max, num_readings)
		side_distances = np.abs(np.sin(thetas) * lidar.ranges)
		ranges_in_front = np.where(side_distances <= 0.19, lidar.ranges, np.inf)
		min_range = np.min(ranges_in_front)

		end_point_ys = np.sin(thetas) * lidar.ranges
		end_point_xs = np.cos(thetas) * lidar.ranges
		end_point_distance_ys = target[1] - end_point_ys
		end_point_distance_xs = target[0] - end_point_xs
		end_point_distances = np.sqrt(np.square(end_point_distance_ys) + np.square(end_point_distance_xs))

		print(f"num_readings: {num_readings}")

		distance_weight = 1000
		weighted_ranges = lidar.ranges * (distance_weight * (1 / end_point_distances))

		if np.sum(weighted_ranges[0:90]) > np.sum(weighted_ranges[90:180]):
			command.angular.z = -6.28
		else:
			command.angular.z = 6.28

		command.linear.x = np.tanh(min_range - 1)

		return command


if __name__ == '__main__':
	rospy.init_node('student_driver', argv=sys.argv)

	driver = StudentDriver()

	rospy.spin()
