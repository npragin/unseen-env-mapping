#!/usr/bin/env python3


import sys
import rospy
import signal
import numpy as np

from controller import RobotController
#Import path_planning and exploring code
from path_planning import dijkstra, open_image, plot_with_path, is_free, get_free_neighbors, convert_image
from exploring import new_find_best_point, find_all_possible_goals, find_highest_concentration_point, find_closest_point, find_best_point, plot_with_explore_points, find_waypoints, find_furthest_point
from helpers import world_to_map, map_to_world
import time

class StudentController(RobotController):
	'''
	This class allows you to set waypoints that the robot will follow.  These robots should be in the map
	coordinate frame, and will be automatially sent to the code that actually moves the robot, contained in
	StudentDriver.
	'''
	def __init__(self):
		super().__init__()
		self._robot_position = None

		self._last_distance_reading = 0
		self._time_since_progress = time.time()
		self._idle_time_allowed = 8

	def distance_update(self, distance):
		'''
		This function is called every time the robot moves towards a goal.  If you want to make sure that
		the robot is making progress towards it's goal, then you might want to check that the distance to
		the goal is generally going down.  If you want to change where the robot is heading to, you can
		make a call to set_waypoints here.  This call will override the current set of waypoints, and the
		robot will start to drive towards the first waypoint in the new list.

		Parameters:
			distance:	The distance to the current goal.
		'''
		if abs(self._last_distance_reading - distance) >= 0.05 and time.time() - self._time_since_progress <= self._idle_time_allowed:
			rospy.loginfo(f"Resetting timer because the distance to the goal has changed by {abs(self._last_distance_reading - distance)} after {time.time() - self._time_since_progress} seconds")
			self._last_distance_reading = distance
			self._time_since_progress = time.time()

	def map_update(self, point, map, map_data):
		'''
		This function is called every time a new map update is available from the SLAM system.  If you want
		to change where the robot is driving, you can do it in this function.  If you generate a path for
		the robot to follow, you can pass it to the driver code using set_waypoints().  Again, this will
		override any current set of waypoints that you might have previously sent.

		Parameters:
			point:		A PointStamped containing the position of the robot, in the map coordinate frame.
			map:		An OccupancyGrid containing the current version of the map.
			map_data:	A MapMetaData containing the current map meta data.
		'''
		rospy.loginfo('Got a map update.')

		# It's possible that the position passed to this function is None.  This try-except block will deal
		# with that.  Trying to unpack the position will fail if it's None, and this will raise an exception.
		# We could also explicitly check to see if the point is None.
		try:
			if self._waypoints is None or len(self._waypoints) == 0 or time.time() - self._time_since_progress > 8:
				self._time_since_progress = time.time()
				# The (x, y) position of the robot can be retrieved like this.
				robot_position_world = (point.point.x, point.point.y)

				self._robot_position = world_to_map(robot_position_world[0], robot_position_world[1], map.info)
				im = np.array(map.data).reshape(map.info.height, map.info.width)
				im_thresh = convert_image(im, wall_threshold=0.8, free_threshold=0.2)
				# rospy.loginfo(f"finding points")
				# points = find_all_possible_goals(im_thresh, map_data)
				# rospy.loginfo(f"points is {points}")
				# # best_point = find_furthest_point(points, self._robot_position)
				# best_point = find_closest_point(points, self._robot_position, map.info)
				# # best_point = find_highest_concentration_point(points, im, map.info)
				# rospy.loginfo(f"best_point was {best_point}")
				# path = dijkstra(im_thresh, self._robot_position, best_point, map_data)

				best_point = new_find_best_point(im_thresh, map_data, self._robot_position)
				path = dijkstra(im_thresh, self._robot_position, best_point, map_data)
				waypoints = find_waypoints(im_thresh, path)
				self.set_waypoints(waypoints)
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
	
	# This will move the robot to a set of fixed waypoints.  You should not do this, since you don't know
	# if you can get to all of these points without building a map first.  This is just to demonstrate how
	# to call the function, and make the robot move as an example.
	#robot_start_loc = controller.get_robot_starting_loc()
	#rospy.loginfo(f"STARTING LOC: {robot_start_loc}")
	#im, im_thresh = open_image("/home/smartw/ros_ws/src/stage_osu/config/simple_rooms.png")
	#waypoints = generate_waypoints(im, im_thresh, robot_start_loc)

	#controller.set_waypoints(((-4,-3),(-4,0),(5,0)))

	# Once you call this function, control is given over to the controller, and the robot will start to
	# move.  This function will never return, so any code below it in the file will not be executed.
	controller.send_points()
