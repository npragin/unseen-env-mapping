#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.ndimage import convolve
import heapq
from math import ceil
import path_planning as path_planning

# ------------------ Plotting candidate exploration points, chosen point, and robot ------------------
def plot_with_explore_points(im_threshhold, zoom=1.0, robot_loc=None, explore_points=None, best_pt=None):
    """
    Plot the map plus, optionally, the robot location, candidate points for exploration,
    and the chosen point for exploration.

    Parameters:
        im_threshhold (numpy.ndarray): The thresholded image of the map
        zoom (float): The zoom level
        robot_loc (tuple): The robot location as an (x, y) pair
        explore_points (list): A list of tuples representing candidate exploration points
                               as (x, y) pairs
        best_pt (tuple): The chosen exploration point as an (x, y) pair
    """

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[0].set_title("original image")
    axs[1].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[1].set_title("threshold image")
    """
    # Used to double check that the is_xxx routines work correctly
    for i in range(0, im_threshhold.shape[1]-1, 10):
        for j in range(0, im_threshhold.shape[0]-1, 2):
            if path_planning.has_free_neighbor(im_thresh, (i, j)):
                axs[1].plot(i, j, '.b')
    """

    # Show original and thresholded image
    if explore_points is not None:
        for p in explore_points:
            axs[1].plot(p[0], p[1], '.b', markersize=2)

    for i in range(0, 2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if best_pt is not None:
            axs[i].plot(best_pt[0], best_pt[1], '*y', markersize=10)
        axs[i].axis('equal')

    for i in range(0, 2):
        # Implements a zoom - set zoom to 1.0 if no zoom
        width = im_threshhold.shape[1]
        height = im_threshhold.shape[0]

        axs[i].set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        axs[i].set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)

# --------------------------------------- Goal point selection ---------------------------------------
def find_frontier_points(im):
    """
    Given a thresholded image of a map, this function returns a list of all frontier
    points. A frontier point is a point in free space adjacent to unseen space.
    
    Parameters:
        im (numpy.ndarray): The thresholded image of the map
    
    Returns:
        numpy.ndarray: List of all frontier points in the map with a shape of (N, 2)
                       where N is the number of frontier points.
    """

    # Create masks for unseen and free points
    unseen_mask = (im == 128)
    free_mask = (im == 255)

    # Create a kernel to check adjacent points
    adjacency_kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Get number of unseen neighbors
    unseen_neighbors = convolve(unseen_mask.astype(int), adjacency_kernel, mode="constant", cval=0)

    # Identify points on the frontier by getting free points with unseen neighbors
    frontier_points_mask = free_mask & (unseen_neighbors > 0)

    # Get coordinates of valid points
    frontier_points = np.argwhere(frontier_points_mask)

    # Swap axes
    frontier_points = frontier_points[:, ::-1]

    return frontier_points

visited = {}
priority_queue = []

def new_find_best_point(map, map_data, robot_loc):
    rospy.loginfo("Starting new_find_best_point")

    # Set the minimum distance from the robot to the goal
    # Using half the lidar range to balance information gain and scan overlap
    lidar_range_in_meters = 8
    lidar_range_in_pixels = ceil(lidar_range_in_meters / map_data.resolution)
    distance_restriction = lidar_range_in_pixels / 2

    free_areas = map != 0

    # Allow processing points too close to the wall if robot is too close to the wall
    process_bad_nodes = np.sum(free_areas[max(0, robot_loc[1] - 1):min(free_areas.shape[0], robot_loc[1] + 2), max(0, robot_loc[0] - 1):min(free_areas.shape[1], robot_loc[0] + 2)]) < 2
    if process_bad_nodes:
        rospy.loginfo("Processing bad nodes.")

    # Initialize data structures for Dijkstra
    # Visited stores (distance from robot, parent node, is node closed) and is indexed using (i,j) tuple
    global visited, priority_queue

    # Updating distance from robot for all points in priority queue
    if len(priority_queue) > 0:
        points = set(p[1] for p in priority_queue if not visited[p[1]][2] and not path_planning.is_wall(map, p[1]))
        distances = path_planning.multi_goal_a_star(map, robot_loc, points)
        priority_queue = [(distance, point) for point, distance in distances.items()]
        heapq.heapify(priority_queue)

    rejected_candidate_goals = []
    furthest_rejected_goal = None

    heapq.heappush(priority_queue, (0, robot_loc))
    visited[robot_loc] = (0, None, False)

    nearest = None
    while priority_queue and nearest is None:
        curr_node_distance, curr_node = heapq.heappop(priority_queue)
        _, curr_node_parent, curr_node_closed = visited[curr_node]

        if curr_node_closed:
            continue

        # TODO: Use a neighbor helper function from path_planning.py
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                neighbor = (curr_node[0] + di, curr_node[1] + dj)

                # Skip nodes that are too close to the wall, unless the robot is, and nodes that are not adjacent to free space
                if (not free_areas[neighbor[1], neighbor[0]] and not process_bad_nodes) or not path_planning.has_free_neighbor(map, neighbor):
                    continue

                if process_bad_nodes and free_areas[neighbor[1], neighbor[0]]:
                    rospy.loginfo("No longer processing bad nodes.")
                    process_bad_nodes = False

                neighbor_distance = curr_node_distance + np.linalg.norm((di, dj))

                if neighbor not in visited:
                    visited[neighbor] = (neighbor_distance, curr_node, False)
                    heapq.heappush(priority_queue, (neighbor_distance, neighbor))
                    
        # Close the node if it's seen or selected as the goal
        if map[curr_node[1], curr_node[0]] == 128:
            if curr_node_distance >= distance_restriction:
                nearest = curr_node
                visited[curr_node] = (curr_node_distance, curr_node_parent, True)
            else:
                if furthest_rejected_goal is None or curr_node_distance > visited[furthest_rejected_goal][0]:
                    furthest_rejected_goal = curr_node
                rejected_candidate_goals.append(curr_node)
        else:
            visited[curr_node] = (curr_node_distance, curr_node_parent, True)

    # If we have to return the furthest rejected goal, remove it from the rejected candidate goals and close it
    if nearest is None and furthest_rejected_goal is not None:
        rejected_candidate_goals.remove(furthest_rejected_goal)
        visited[furthest_rejected_goal] = (visited[furthest_rejected_goal][0], visited[furthest_rejected_goal][1], True)

    # Add the rejected candidate goals back to the priority queue to reconsider them later
    for point in rejected_candidate_goals:
        heapq.heappush(priority_queue, (visited[point][0], point))

    return nearest if nearest is not None else furthest_rejected_goal

    # Leaving this in case we can find a way to construct a path from this algorithm
    path = []
    current = nearest
    while current:
        current_x_in_space = current[0] * map_data.resolution + map_data.origin.position.x
        current_y_in_space = current[1] * map_data.resolution + map_data.origin.position.y
        path.append((current_x_in_space, current_y_in_space))
        current = visited[current][1]

    path.reverse()
    return path

def find_closest_point(candidate_points, robot_loc, min_distance=0):
    """
    Returns the point from a list of candidate points closest to the robot's location
    and, optionally, a minimum distance from the robot's location.

    Parameters:
        candidate_points (numpy.ndarray): List of (x, y) pairs of candidate points
        robot_loc (tuple): (x, y) pair representing the robot's current position
        min_distance (float): The minimum distance for the selected point to be from the
                              robot.

    Returns:
        tuple: (x, y) pair of the closest point to the robot's location, such that the
               point is not directly beneath the robot
    """
    distances = np.linalg.norm(candidate_points - robot_loc, axis=1)
    # Replace distances < min_distance with np.inf
    masked_distances = np.where(distances >= min_distance, distances, np.inf)
    return candidate_points[np.argmin(masked_distances)]

def find_furthest_point(candidate_points, robot_loc):
    """
    Returns the point from a list of candidate points furthest to the robot's location.

    Parameters:
        candidate_points (numpy.ndarray): List of (x, y) pairs of candidate points
        robot_loc (tuple): (x, y) pair representing the robot's current position

    Returns:
        tuple: (x, y) pair of the furthest point to the robot's location
    """

    distances = np.linalg.norm(candidate_points - robot_loc, axis=1)
    furthest_idx = np.argmax(distances)
    return candidate_points[furthest_idx]

def find_highest_concentration_point(possible_points, im, map_data, radius=0.5):
    radius_pixels = 10 #int(radius / map_data.resolution)

    # Create a binary image of all target points (where im == 128)
    target_points = np.zeros_like(im, dtype=float)
    target_points[im == 128] = 1

    # Create circular kernel
    y, x = np.ogrid[-radius_pixels:radius_pixels+1, -radius_pixels:radius_pixels+1]
    kernel = x*x + y*y <= radius_pixels*radius_pixels
    kernel = kernel.astype(float)

    # Apply convolution to count nearby points
    concentration_map = convolve(target_points, kernel, mode='constant', cval=0.0)

    # Create a mask of possible points
    possible_points = np.array(list(possible_points))
    points_mask = np.zeros_like(im, dtype=bool)
    points_mask[possible_points[:, 0].astype(int), possible_points[:, 1].astype(int)] = True

    # Mask the concentration map to only look at possible points
    masked_concentration = np.where(points_mask, concentration_map, -1)

    # Find the point with highest concentration
    max_idx = np.unravel_index(np.argmax(masked_concentration), masked_concentration.shape)
    rospy.loginfo(f"max_idx type: {type(max_idx)}, {max_idx}")
    return max_idx

def calculate_vector(point1, point2):
    """Calculate the vector from point1 to point2."""
    return point2[0] - point1[0], point2[1] - point1[1]

def generate_waypoints(path):
    """ Place waypoints along the path
    @param path - the initial path
    @ return - a new path"""

    if len(path) < 3:
        return path[1:]

    waypoints = []

    for i in range(1, len(path) - 1):
        # Calculate direction vectors between consecutive points
        v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        
        # Check if vectors are not collinear using cross product
        # Add point to the path if the path's direction changes at that point
        if np.cross(v1, v2) != 0:
            waypoints.append(path[i])

    # Append the goal point
    waypoints.append(path[-1])

    return waypoints
