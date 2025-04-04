#!/usr/bin/env python3


import rospy
import sys
import numpy as np

from math import atan2, tanh, sqrt, pi, ceil, degrees

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

import actionlib
import tf

from lab2.msg import NavTargetAction, NavTargetResult, NavTargetFeedback


class Driver:
	def __init__(self, position_source, threshold=0.1):
		# Goal will be set later. The action server will set the goal; you don't set it directly
		self.goal = None
		self.threshold = threshold

		self.transform_listener = tf.TransformListener()

		# Publisher before subscriber
		self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		self.target_pub = rospy.Publisher('current_target', Marker, queue_size=1)

		# Subscriber after publisher
		self.sub = rospy.Subscriber('base_scan', LaserScan, self._callback, queue_size=1)

		# Action client
		self.action_server = actionlib.SimpleActionServer('nav_target', NavTargetAction, execute_cb=self._action_callback, auto_start=False)
		self.action_server.start()

	@classmethod
	def zero_twist(cls):
		"""This is a helper class method to create and zero-out a twist"""
		command = Twist()
		command.linear.x = 0.0
		command.linear.y = 0.0
		command.linear.z = 0.0
		command.angular.x = 0.0
		command.angular.y = 0.0
		command.angular.z = 0.0

		return command

	# Respond to the action request.
	def _action_callback(self, goal):
		""" This gets called when an action is received by the action server
		@goal - this is the new goal """
		rospy.loginfo(f'Got an action request for ({goal.goal.point.x:.2f}, {goal.goal.point.y:.2f})')

		# Set the goal.
		self.goal = goal.goal

		# Build a marker for the goal point
		#   - this prints out the green dot in RViz (the current goal)
		marker = Marker()
		marker.header.frame_id = goal.goal.header.frame_id
		marker.header.stamp = rospy.Time.now()
		marker.id = 0
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.position = goal.goal.point
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0		
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.3
		marker.scale.y = 0.3
		marker.scale.z = 0.3
		marker.color.r = 0.0
		marker.color.g = 1.0
		marker.color.b = 0.0
		marker.color.a = 1.0

		# Wait until we're at the goal.  Once we get there, the callback that drives the robot will set self.goal
		# to None.
		while self.goal:
			self.target_pub.publish(marker)
			rospy.sleep(0.1)

		rospy.loginfo('Action completed')

		# Build a result to send back
		result = NavTargetResult()
		result.success.data = True

		self.action_server.set_succeeded(result)

		# Get rid of the marker
		marker.action = Marker.DELETE
		self.target_pub.publish(marker)

	def _callback(self, lidar):
		# If we have a goal, then act on it, otherwise stay still
		if self.goal:
			# Update the timestamp on the goal and figure out where it it now in the base_link frame.
			self.goal.header.stamp = rospy.Time.now()
			target = self.transform_listener.transformPoint('base_link', self.goal)

			rospy.loginfo(f'Target: ({target.point.x:.2f}, {target.point.y:.2f})')

			# Are we close enough?  If so, then remove the goal and stop
			distance = sqrt(target.point.x ** 2 + target.point.y ** 2)

			feedback = NavTargetFeedback()
			feedback.distance.data = distance
			self.action_server.publish_feedback(feedback)

			if distance < self.threshold:
				self.goal = None
				command = Driver.zero_twist()
			else:
				command = self.get_twist((target.point.x, target.point.y), lidar)
		else:
			command = Driver.zero_twist()

		self.cmd_pub.publish(command)

	# This is the function that controls the robot.
	#
	# Inputs:
	# 	target:	a tuple with the (x, y) coordinates of the target point, in the robot's coordinate frame (base_link).
	# 			x-axis is forward, y-axis is to the left.
	# 	lidar:	a LaserScan message with the current data from the LiDAR.  Use this for obstacle avoidance.
	#           This is the same as your go and stop code
	def old_get_twist(self, target, lidar):
		w = 0.76
		command = Driver.zero_twist()
		thetas = np.linspace(lidar.angle_min, lidar.angle_max, len(lidar.ranges))
		ranges = np.array(lidar.ranges)

		# Get the closest obstacle in front of us and within 2 meters
		in_front_idx = np.where((ranges * np.abs(np.sin(thetas)) <= w/2) & (ranges < 2))[0]
		# print(in_front_idx)
		# If nothing is in front of us within 2 meters
		if len(in_front_idx) == 0:
			#  Step 1) Calculate the angle the robot has to turn to in order to point at the target
			target_angle = np.arctan2(target[1], target[0])
			target_distance = np.linalg.norm(np.array(target))

			#  Step 2) Set your speed based on how far away you are from the target, as before
			command.linear.x = target_distance / 2
			command.angular.z = target_angle * 0.75
		else:
			left = 0
			right = 0

			for r, t in zip(ranges, thetas):
				if t < 0:
					right += r
				else:
					left += r
			
			print("Left: ", left, "Right: ", right)

			if right > left:
				command.angular.z = right / (left + right) * -0.5
			else:
				command.angular.z = left / (left + right) * 0.5
			
			command.linear.x = 0.1 if np.min(ranges[in_front_idx]) < 0.25 else np.min(ranges[in_front_idx]) * 0.1

		return command

	def get_twist(self, target, lidar):
		w = 0.38 # Robot's width
		l = 0.44
		command = Driver.zero_twist() 
		thetas = np.linspace(lidar.angle_min, lidar.angle_max, len(lidar.ranges))
		ranges = np.array(lidar.ranges)

		obstacle_threshold = 1.5 + (l / 2)
		obstacles_in_front_idx = np.where((ranges * np.abs(np.sin(thetas)) <= w/2) & (ranges < obstacle_threshold))[0]

		if (len(obstacles_in_front_idx) == 0):
			rospy.loginfo("Driving to target.")
			target_angle = atan2(target[1], target[0])
			target_distance = np.linalg.norm(np.array(target))

			command.linear.x = target_distance / 2
			command.angular.z = target_angle * 0.75
			return command
		else:
			rospy.loginfo("Avoiding obstacles.")

			obstacle_distance = np.min(ranges[obstacles_in_front_idx]) - l / 2

			angle_of_concern = 2 * abs(np.arctan(w / 2 / obstacle_threshold))
			angle_per_scan = ((lidar.angle_max - lidar.angle_min) / len(lidar.ranges))
			num_scans_of_concern = ceil(angle_of_concern / angle_per_scan)

			cones = np.lib.stride_tricks.sliding_window_view(ranges, num_scans_of_concern)
			safe_cones_idx = np.nonzero(np.all(cones > obstacle_threshold, axis=1))[0]

			if len(safe_cones_idx) == 0:
				rospy.loginfo("SHOULD DO A 180")
				return command # TODO: Do a 180

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
	rospy.init_node('driver', argv=sys.argv)

	driver = Driver('odom')

	rospy.spin()
