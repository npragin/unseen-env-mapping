#!/usr/bin/env python3


import sys
import rospy
import numpy as np

from controller import RobotController
#Import path_planning and exploring code
from path_planning import a_star, convert_map_to_configuration_space
from exploring import new_find_best_point, generate_waypoints, find_frontier_points, find_highest_information_gain_point, find_closest_point, find_furthest_point
from helpers import world_to_map, save_map_as_image
import time
from math import ceil

class StudentController(RobotController):
	'''
	This class sets waypoints that the robot will follow. These waypoints should be in
	the map coordinate frame, and will be automatially sent to the code that moves the
	robot, contained in StudentDriver.
	'''
	def __init__(self):
		super().__init__()
		self._robot_position_map = None

		self._robot_width_in_meters = 0.38
		self._robot_height_in_meters = 0.44
		self._robot_diagonal_length_in_meters = np.linalg.norm((self._robot_width_in_meters, self._robot_height_in_meters))

		# We want to keep track of progress towards the goal to understand if we are stuck
		self._last_distance_reading = 0
		self._idle_time_allowed = 8
		self._distance_threshold = 0.2

	def distance_update(self, distance):
		'''
		Updates the robot's progress tracking as it moves toward its current goal point.
		
		Checks if meaningful progress has been made by comparing the change in distance
		against a threshold, tracks how long the robot has been idle (not making progress),
		and updates progress tracking only when meaningful progress is made within the
		allowed idle time

		Parameters:
			distance:	The distance to the current goal.
		'''
		distance_traveled = abs(self._last_distance_reading - distance)
		meaningful_progress_made = distance_traveled >= self._distance_threshold
		idle_time = time.time() - self._time_since_progress

		if meaningful_progress_made and idle_time <= self._idle_time_allowed:
			self._last_distance_reading = distance
			self._time_since_progress = time.time()

	def map_update(self, point, map, map_data):
		'''
		Processes new map updates from the SLAM system and generates waypoints for the robot.
		
		This function is called whenever a new map update is available and generates new
		waypoints if we have no pending waypoints or the robot is stuck.

		The function converts the robot's position to map coordinates, creates a 
		configuration space representation of the map, finds an optimal goal point, 
		generates a path to that goal using A*, and creates waypoints from that path.
		If no valid goal point is found, it saves the final map and shuts down all nodes.

		Parameters:
			point (PointStamped):	The position of the robot, in the world coordinate frame.
			map (OccupancyGrid):	The current version of the map.
			map_data (MapMetaData):	The current map metadata.
		'''
		rospy.loginfo('Got a map update.')

		# It's possible that the position (point) passed to this function is None.
		# This try-except block will deal with that.
		try:
			# Only generate a goal point if we don't have any waypoints or if we are stuck
			if self._waypoints is None or len(self._waypoints) == 0 or time.time() - self._time_since_progress > 8:
				# Update time since we last made progress
				self._time_since_progress = time.time()

				# The (x, y) position of the robot in the world
				robot_position_world = (point.point.x, point.point.y)

				# Convert the robot's position to the map coordinate frame
				self._robot_position_map = world_to_map(robot_position_world[0], robot_position_world[1], map.info)

				# Convert the map to a 2D numpy array
				im = np.array(map.data).reshape(map.info.height, map.info.width)

				# Convert the map to a threshold image in the configuration space using
				# the robot's diagonal length to ensure the robot can always safely
				# rotate in place
				robot_diagonal_length_in_pixels = ceil(self._robot_diagonal_length_in_meters / map_data.resolution)
				im_thresh = convert_map_to_configuration_space(im, 0.8, 0.2, robot_diagonal_length_in_pixels)

				'''
				Stale code that chose a goal point based on an information-theoretic
				approach or a convolution-based geometric approach. Kept for reference.

				rospy.loginfo(f"Selecting goal point")
				candidate_points = find_frontier_points(im_thresh)

				# goal_point = find_furthest_point(candidate_points, self._robot_position_map)
				# goal_point = find_closest_point(candidate_points, self._robot_position_map, 0)
				goal_point = find_highest_concentration_point(candidate_points, im, map.info)
				'''

				# Select the goal point using a BFS-based geometric approach
				goal_point = new_find_best_point(im_thresh, map_data, self._robot_position_map)

				# If no goal point is found, we are done and save the map as an image
				if goal_point is None:
					rospy.logerr(f"Finished in {rospy.get_time()} seconds.")
					if save_map_as_image(im):
						rospy.logerr(f"Saved map as image.")
					else:
						rospy.logerr(f"Failed to save map as image.")
					self._shutdown_all_nodes()

				# Generate a path to the goal point
				path = a_star(im_thresh, self._robot_position_map, goal_point, map_data)

				# Chop the path into waypoints
				waypoints = generate_waypoints(im_thresh, path)
				self.set_waypoints(waypoints)

				# Update time since we last made progress, again
				# We need this because generating a goal point takes time and we don't
				# want to generate a goal while we generating the previous one
				self._time_since_progress = time.time()
		except Exception as e:
			import traceback
			rospy.logerr(f"Error in map_update: {e} \n {traceback.format_exc()}")

			rospy.loginfo('No odometry information.')

	def get_robot_starting_loc(self):
		while not rospy.is_shutdown():
			if self._odom:
				position = self._odom.pose.pose.position
				return (position.x, position.y)
			rospy.sleep(0.1)

if __name__ == '__main__':
	# Initialize the node.
	rospy.init_node('student_controller', argv=sys.argv)

	# Start the controller.
	controller = StudentController()

	# Once you call this function, control is given over to the controller, and the robot will start to
	# move.  This function will never return, so any code below it in the file will not be executed.
	controller.send_points()
