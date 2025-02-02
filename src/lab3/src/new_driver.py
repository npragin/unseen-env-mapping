#!/usr/bin/env python3


import rospy
import sys

from math import sqrt, pi
import time
import numpy as np

from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan

import actionlib
import tf

from lab2.msg import NavTargetAction, NavTargetResult, NavTargetFeedback

class Driver:
	def __init__(self, position_source, threshold=0.1):
		self._target_point = None
		self._threshold = threshold

		# Used to rotate the robot in-place, do a 360 to start to understand surroundings
		# Set self._rotate_count to positive to rotate left, negative to rotate right
		self._360_ROTATE_COUNT = 39
		self._rotate_count = self._360_ROTATE_COUNT

		self.transform_listener = tf.TransformListener()

		self._cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
		self.target_pub = rospy.Publisher('current_target', Marker, queue_size=1)

		self._lidar_sub = rospy.Subscriber('base_scan', LaserScan, self._lidar_callback, queue_size=10)

		# Action client
		self._action_server = actionlib.SimpleActionServer('nav_target', NavTargetAction, execute_cb=self._action_callback, auto_start=False)
		self._action_server.start()

		self._time_of_last_target_point = time.time()

	@classmethod
	def zero_twist(cls):
		"""
		Creates a Twist object with all linear and angular velocities set to zero.

		Returns:
			command (Twist): A Twist object with zeroed linear and angular velocities.
		"""
		t = Twist()
		t.linear.x = 0.0
		t.linear.y = 0.0
		t.linear.z = 0.0
		t.angular.x = 0.0
		t.angular.y = 0.0
		t.angular.z = 0.0

		return t

	def _rotate(self):
		"""
		Generates a rotation command for the driver.

		This method creates a rotation command with an angular velocity based on _rotate_count.
		If _rotate_count is not zero, the angular velocity is set to 2 * pi in the direction
		indicated by the sign of _rotate_count. _rotate_count is then decremented by 1.

		Returns:
			command (Twist): A Driver command object with the specified angular velocity.
		"""
		command = Driver.zero_twist()

		if self._rotate_count != 0:
			command.angular.z = 2 * pi * np.sign(self._rotate_count)
			self._rotate_count -= 1 * np.sign(self._rotate_count)

		return command

	def rotate_360(self):
		"""
		Sets _rotate_count to rotate the robot 360 degrees if it isn't already rotating.
		"""
		if self._rotate_count == 0:
			self._rotate_count = self._360_ROTATE_COUNT

	def rotate_180(self):
		"""
		Sets _rotate_count to rotate the robot 360 degrees if it isn't already rotating.
		"""
		if self._rotate_count == 0:
			self._rotate_count = int(self._360_ROTATE_COUNT / 2)

	def rotate_90_left(self):
		"""
		Sets _rotate_count to rotate the robot 90 degrees leftward if it isn't already
		rotating.
		"""
		if self._rotate_count == 0:
			self._rotate_count = int(self._360_ROTATE_COUNT / 4)

	def rotate_90_right(self):
		"""
		Sets _rotate_count to rotate the robot 90 degrees rightward if it isn't already
		rotating.
		"""
		if self._rotate_count == 0:
			self._rotate_count = -1 * int(self._360_ROTATE_COUNT / 4)

	def _get_rotate_count(self, angle):
		"""
		Returns the value to set _rotate_count to rotate the robot to face the specified
		angle.

		Parameters:
			angle (float): The angle to rotate the robot to face, in radians.

		Returns:
			rotate_count (int): The value to set _rotate_count to rotate the robot to face
								the specified angle
		"""
		return round(angle / (2 * pi) * self._360_ROTATE_COUNT) * np.sign(angle)

	# Respond to the action request.
	def _action_callback(self, goal):
		rospy.loginfo(f'Got an action request for ({goal.goal.point.x:.2f}, {goal.goal.point.y:.2f})')

		result = NavTargetResult()

		self._target_point = goal.goal

		# Build a marker for the goal point
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

		# Loops while the goal is valid, stops if preempted
		rate = rospy.Rate(100)
		while self._target_point:
			if self._action_server.is_preempt_requested():
				self._target_point = None
				result.success.data = False
				self._action_server.set_succeeded(result)
				return

			self.target_pub.publish(marker)
			rate.sleep()

		# If not preempted, set the result to success
		result.success.data = True
		self._action_server.set_succeeded(result)

	def _lidar_callback(self, lidar):
		# If we are rotating, don't do anything else
		if self._rotate_count != 0:
			command = self._rotate()
		elif self._target_point:
			self._time_of_last_target_point = time.time()
			# NOTE: Added a delay to account for the time it takes to get the lidar data
			self._target_point.header.stamp = rospy.Time.now() - rospy.Duration(0.2)
			try:
				# Transform the target point into the robot's coordinate frame
				target = self.transform_listener.transformPoint('base_link', self._target_point)

				x = target.point.x
				y = target.point.y
				distance = sqrt(x ** 2 + y ** 2)

				feedback = NavTargetFeedback()
				feedback.distance.data = distance
				self._action_server.publish_feedback(feedback)

				# If we've reached the target point, mark it as complete, otherwise drive to it
				if self.close_enough_to_waypoint(distance, (target.point.x, target.point.y), lidar):
					self._target_point = None
					command = Driver.zero_twist()
				else:
					command = self.get_twist((target.point.x, target.point.y), lidar)

			except Exception as e:
				import traceback
				rospy.logerr(f"Error in lidar_callback {e} \n {traceback.format_exc()}")
				return
		else:
			rospy.logerr("NO TARGET POINT")
			# Rotate while we wait for a new target point
			# Only rotate if we haven't had a target in the last second to avoid spinning
			# between waypoints, which can be disorienting. We only want to rotate
			# while generating a new goal point.
			if time.time() - self._time_of_last_target_point > 1:
				self._rotate_count += 1
				command = self._rotate()
			else:
				command = Driver.zero_twist()
		self._cmd_pub.publish(command)

	def close_enough_to_waypoint(self, distance, target, lidar):
		"""
		This is an abstract method that will be overridden by student_driver
		"""
		raise NotImplemented('close_enough_to_waypoint() not implemented')

	def get_twist(self, target, lidar):
		"""
		This is an abstract method that will be overridden by student_driver
		"""
		raise NotImplemented('get_twist() not implemented')


if __name__ == '__main__':
	rospy.init_node('driver', argv=sys.argv)

	driver = Driver('odom')

	rospy.spin()
