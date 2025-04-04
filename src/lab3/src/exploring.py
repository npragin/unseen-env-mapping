#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.ndimage import convolve
import heapq
from math import ceil
import path_planning as path_planning

# ------------------ Plotting candidate exploration points, chosen point, and robot ------------------
def plot_with_explore_points(map, zoom=1.0, robot_loc=None, candidate_points=None, goal=None):
    """
    Plot the map plus, optionally, the robot location, candidate points for exploration,
    and the chosen point for exploration.

    Parameters:
        map (numpy.ndarray): A thresholded image of the map where:
                                - 0 represents an obstacle
                                - 128 represents unseen space
                                - 255 represents free space
        zoom (float): The zoom level
        robot_loc (tuple): The robot location as an (x, y) pair
        candidate_points (list): A list of tuples representing candidate exploration
                                 points as (x, y) pairs
        goal (tuple): The goal as an (x, y) pair
    """

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(map, origin='lower', cmap="gist_gray")
    axs[0].set_title("original image")
    axs[1].imshow(map, origin='lower', cmap="gist_gray")
    axs[1].set_title("threshold image")
    """
    # Used to double check that the is_xxx routines work correctly
    for i in range(0, im_threshhold.shape[1]-1, 10):
        for j in range(0, im_threshhold.shape[0]-1, 2):
            if path_planning.has_free_neighbor(map, (i, j)):
                axs[1].plot(i, j, '.b')
    """

    # Show original and thresholded image
    if candidate_points is not None:
        for p in candidate_points:
            axs[1].plot(p[0], p[1], '.b', markersize=2)

    for i in range(0, 2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if goal is not None:
            axs[i].plot(goal[0], goal[1], '*y', markersize=10)
        axs[i].axis('equal')

    for i in range(0, 2):
        # Implements a zoom - set zoom to 1.0 if no zoom
        width = map.shape[1]
        height = map.shape[0]

        axs[i].set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        axs[i].set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)

# --------------------------------------- Goal point selection ---------------------------------------
def convolutional_frontier_detection(map):
    """
    Given a thresholded image of a map, this function returns a list of all frontier
    points. A frontier point is a point in free space adjacent to unseen space.
    
    Parameters:
        map (numpy.ndarray): A thresholded image of the map where:
                                - 0 represents an obstacle
                                - 128 represents unseen space
                                - 255 represents free space
    
    Returns:
        numpy.ndarray: List of all frontier points in the map with a shape of (N, 2)
                       where N is the number of frontier points.
    """

    # Create masks for unseen and free points
    unseen_mask = (map == 128)
    free_mask = (map == 255)

    # Create a kernel to check adjacent points
    adjacency_kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    # Get number of unseen neighbors
    unseen_neighbors = convolve(unseen_mask.astype(int), adjacency_kernel, mode="constant")

    # Identify points on the frontier by getting free points with unseen neighbors
    frontier_points_mask = free_mask & (unseen_neighbors > 0)

    # Get coordinates of valid points
    frontier_points = np.argwhere(frontier_points_mask)

    # Swap axes
    frontier_points = frontier_points[:, ::-1]

    return frontier_points

is_closed = {}
priority_queue = []

def expanding_wavefront_frontier_detection(map, robot_loc, distance_restriction=0):
    """
        Returns the closest frontier point on the map to the robot with an optional
        minimum distance.

        This algorithm uses the Expanding Wavefront Algorithm proposed by Quin, et al.
        leveraging global data structures to guarantee completeness and prevent
        processing the same point twice.

        Parameters:
            map (numpy.ndarray): A thresholded image of the map where:
                                    - 0 represents an obstacle
                                    - 128 represents unseen space
                                    - 255 represents free space
            robot_loc (tuple): (x, y) pair representing the robot's current position
            distance_restriction (float): Optional minimum distance between the selected
                                          frontier point and the robot.
        Returns:
            tuple: Returns the closest frontier point as an (x, y) pair on the map to
                   robot_loc, at least as far from the robot as distance_restriction. 
    """
    # Initialize data structures for Dijkstra
    # is_closed is indexed using (x, y) tuple
    global is_closed, priority_queue
    candidate_goals = []
    furthest_rejected_goal = None
    furthest_rejected_goal_dist = 0
    goal = None

    # Update distance from robot to all points in priority queue if they are free and not closed
    if len(priority_queue) > 0:
        points = set(p[1] for p in priority_queue if not is_closed[p[1]] and path_planning.is_free(map, p[1]))
        distances = path_planning.multi_goal_a_star(map, robot_loc, points)
        priority_queue = [(distance, point) for point, distance in distances.items()]
        heapq.heapify(priority_queue)

    # Add robot_loc to the priority queue to start the search there if it isn't closed
    robot_loc_closed = is_closed.get(robot_loc, False)
    if not robot_loc_closed:
        heapq.heappush(priority_queue, (0, robot_loc))
        is_closed[robot_loc] = False

    while priority_queue and goal is None:
        curr_node_distance, curr_node = heapq.heappop(priority_queue)
        curr_node_closed = is_closed[curr_node]

        # If this node is closed, skip it
        if curr_node_closed:
            continue

        # If this node is a frontier point, check if the distance restriction is
        # satisfied and set it as the goal if yes. Save the furthest goal to use if no
        # point satisfies the distance restriction. Don't process points with unseen
        # neighbors to avoid adding unseen nodes to the priority queue
        if path_planning.has_unseen_neighbor(map, curr_node):
            if curr_node_distance >= distance_restriction:
                goal = curr_node
            elif furthest_rejected_goal is None or curr_node_distance > furthest_rejected_goal_dist:
                    furthest_rejected_goal = curr_node
                    furthest_rejected_goal_dist = curr_node_distance
            candidate_goals.append(curr_node)
            continue
        else:
            is_closed[curr_node] = True

        # Add neighbors to the priority queue
        for neighbor, neighbor_cost in path_planning.get_free_neighbors_with_cost(map, curr_node):
            neighbor_distance = curr_node_distance + neighbor_cost

            if neighbor not in is_closed:
                is_closed[neighbor] = False
                heapq.heappush(priority_queue, (neighbor_distance, neighbor))

    # Add the rejected candidate goals back to the priority queue to reconsider them later
    # An arbitrary distance value is used because the distance will be recalculated with
    # the updated robot location.
    for point in candidate_goals:
        heapq.heappush(priority_queue, (0, point))

    return goal if goal is not None else furthest_rejected_goal

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

def find_highest_information_gain_point(candidate_points, map, radius):
    """
    Returns the point from a list of candidates that has the highest concentration of
    unexplored pixels within a given radius.

    **WARNING**: This function will count unseen points through walls

    Parameters:
        candidate_points (numpy.ndarray): List of (x, y) pairs of candidate points
        map (numpy.ndarray): A thresholded image of the map where:
                                - 0 represents an obstacle
                                - 128 represents unseen space
                                - 255 represents free space
        radius (int): Radius in pixels to check for unseen points around each candidate
                      point

    Returns:
        tuple: (y, x) coordinates of the point with highest number of unexplored points
               within its radius
    """
    # Create mask of unseen points
    unseen_points = map == 128

    # Create circular kernel
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (x ** 2 + y ** 2 <= radius ** 2).astype(float)

    # Convolve to calculate information gain at each point
    information_gain_map = convolve(unseen_points, kernel, mode='constant')

    # Create a mask of candidate points
    candidate_points_mask = np.zeros_like(map, dtype=bool)
    candidate_points_mask[candidate_points[:, 1].astype(int), candidate_points[:, 0].astype(int)] = True

    # Mask the information gain map to only consider candidate points
    masked_ig_map = np.where(candidate_points_mask, information_gain_map, -1)

    # Find the point with highest information gain
    max_ig_point = np.unravel_index(np.argmax(masked_ig_map), masked_ig_map.shape)

    return max_ig_point

def calculate_vector(point1, point2):
    """
    Calculate the vector from point1 to point2

    Parameters:
        point1 (tuple): A point as an (x, y) pair
        point2 (tuple): A point as an (x, y) pair

    Returns:
        tuple: The displacement vector from point1 to point2 as a (dx, dy) pair
    """
    return (point2[0] - point1[0], point2[1] - point1[1])

def generate_waypoints(map, path):
    """
    Generate a lower resolution path using a Line-of-Sight algorithm

    Parameters:
        path (numpy.ndarray): A list of points as (x, y) pairs representing the initial
                              path. The first point is assumed to be the start location.

    Returns:
        list: A simplified path containing only the points needed to maintain line of
              sight with the next point. Includes the start and the goal points.
    """
    if len(path) <= 2:
        return path

    # Include the start point
    waypoints = [path[0]]

    current_index = 0
    while current_index < len(path) - 1:
        # Find the furthest point that has line of sight from the current waypoint using
        # the next point as the default
        furthest_index = current_index + 1
        for j in range(len(path) - 1, current_index, -1):
            if has_line_of_sight(map, path[current_index], path[j]):
                furthest_index = j
                break

        waypoints.append(path[furthest_index])
        current_index = furthest_index

    return waypoints

def has_line_of_sight(map, start, end):
    """
    Check if there's a clear line of sight from start to end using Bresenham's algorithm
    
    Parameters:
        map (numpy.ndarray): The thresholded image of the map
        start (tuple): (x, y) pair representing the start of the line
        end (tuple): (x, y) pair representing the end of the line
        
    Returns:
        bool: True if there's line of sight, False otherwise
    """
    x0, y0 = start
    x1, y1 = end

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0

    while True:
        if path_planning.is_wall(map, (x, y)):
            return False

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return True
