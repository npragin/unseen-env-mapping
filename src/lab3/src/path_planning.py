#!/usr/bin/env python3

import rospy
import numpy as np
import heapq
import cv2

from helpers import save_map_as_debug_image


# ------------------ Plotting path, robot, and goal location ------------------
def plot_with_path(map, zoom=1.0, robot_loc=None, goal=None, path=None):
    """
    Plot the map and, optionally, the robot location, goal location, and proposed path

    Parameters:
        map (numpy.ndarray): The thresholded image of the map
        zoom (float): The zoom level
        robot_loc (tuple): The robot location as an (x, y) pair
        goal (tuple): The goal location as an (x, y) pair
        path (list): A list of tuples representing the path as (x, y) pairs
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
        for j in range(0, im_threshhold.shape[0]-1, 10):
            if is_wall(map, (i, j)):
                axs[1].plot(i, j, '.b')
    """

    # Double checking lower left corner
    axs[1].plot(10, 5, 'xy', markersize=5)

    # Show original and thresholded image
    for i in range(0, 2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if goal is not None:
            axs[i].plot(goal[0], goal[1], '*g', markersize=10)
        if path is not None:
            for p, q in zip(path[0:-1], path[1:]):
                axs[i].plot([p[0], q[0]], [p[1], q[1]], '-y', markersize=2)
                axs[i].plot(p[0], p[1], '.y', markersize=2)
        axs[i].axis('equal')

    for i in range(0, 2):
        # Implements a zoom - set zoom to 1.0 if no zoom
        width = map.shape[1]
        height = map.shape[0]

        axs[i].set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        axs[i].set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)


# -------------- Thresholded image pixel identification ---------------
def is_wall(map, loc):
    """
    Returns true if the location in the map is an obstacle

    Parameters:
        map (numpy.ndarray): The thresholded image
        loc (tuple): The coordinate as an (x, y) pair
    """
    if map[loc[1], loc[0]] == 0:
        return True
    return False

def is_unseen(map, loc):
    """
    Returns true if the location in the map is unseen

    Parameters:
        map (numpy.ndarray): The thresholded image
        loc (tuple): The coordinate as an (x, y) pair
    """
    if map[loc[1], loc[0]] == 128:
        return True
    return False

def is_free(map, loc):
    """
    Returns true if the location in the map is free space

    Parameters:
        map (numpy.ndarray): The thresholded image
        loc (tuple): The coordinate as an (x, y) pair
    """
    if map[loc[1], loc[0]] == 255:
        return True
    return False


# -------------- Occupancy grid to threshold image conversion ---------------
def convert_image(occupancy_grid, wall_threshold, free_threshold):
    """
    Convert the OccupancyGrid to a thresholded image. Any points that have a confidence
    level below what is required by the provided thresholds will be marked as unseen
    space. The threshold image will have 3 values:
        - 0 representing an obstacle
        - 128 representing unseen space
        - 255 representing free space

    Parameters:
        occupancy_grid (numpy.ndarray): An OccupancyGrid's data member resized to a 2D
                                        array
        wall_threshold (float): The threshold value for identifying walls.
                                Expected to be in the range of [free_threshold, 1], where
                                the float value indicates the certainty required to mark
                                a point in space as an obstacle
        free_threshold (float): The threshold value for identifying free space.
                                Expected to be in the range of [0, wall_threshold], where
                                (1 - the float value) indicates the certainty required to
                                mark a point in space as free

    Returns:
        numpy.ndarray:  A thresholded image with values 0, 128, and 255
                        The dimensions will match the occupancy_grid parameter
    """
    # Assume all is unseen
    im_ret = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1]), dtype='uint8') + 128

    # Set cells with confidence above threshold values to free or obstacle
    im_ret[occupancy_grid > wall_threshold] = 0
    im_ret[(occupancy_grid < free_threshold) & (occupancy_grid != -1)] = 255
    return im_ret

def convert_map_to_configuration_space(occupancy_grid, wall_threshold, free_threshold, robot_dimension):
    """
    Convert the OccupancyGrid to a thresholded image in the configuration space.
    Any points that have a confidence level below what is required by the provided
    thresholds will be marked as unseen space. The threshold image will have 3 values:
        - 0 represents an obstacle
        - 128 represents unseen space
        - 255 represents free space

    Parameters:
        occupancy_grid (numpy.ndarray): An OccupancyGrid's data member resized to a 2D
                                        array
        wall_threshold (float): The threshold value for identifying walls.
                                Expected to be in the range of [free_threshold, 1], where
                                the float value indicates the certainty required to mark
                                a point in space as an obstacle
        free_threshold (float): The threshold value for identifying free space.
                                Expected to be in the range of [0, wall_threshold], where
                                (1 - the float value) indicates the certainty required to
                                mark a point in space as free
        robot_dimension (int): The dimension of the robot in pixels. Used to inflate
                               obstacles by (robot_dimension / 2) to ensure the robot's
                               center point can safely navigate the configuration space

    Returns:
        numpy.ndarray:  A thresholded image with values 0, 128, and 255
                        The dimensions will match the map parameter
    """

    # Assume all is unseen
    im_ret = np.zeros((occupancy_grid.shape[0], occupancy_grid.shape[1]), dtype='uint8') + 128

    # Set cells with confidence above threshold values to free
    im_ret[(occupancy_grid < free_threshold) & (occupancy_grid != -1)] = 255

    # Use mask indicating walls to dilate the obstacles by robot_dimension / 2
    wall_mask = (occupancy_grid > wall_threshold).astype(np.uint8)
    kernel = np.ones((robot_dimension, robot_dimension), dtype='uint8')
    dilated = cv2.dilate(wall_mask, kernel, iterations=1)
    im_ret[dilated > 0] = 0
    
    return im_ret


# ----------------------- Getting neighbors -----------------------
def get_neighbors(map, loc):
    """
    Returns a list of neighbors for a given location in the map

    Parameters:
        map (numpy.ndarray): The image
        loc (tuple): The location as a tuple (x, y)
    
    Returns:
        list: A list of tuples representing the neighbors as (x, y)
    """
    i, j = loc
    neighbors = [
        (i-1, j),
        (i+1, j),
        (i, j-1),
        (i, j+1),
        (i-1, j-1),
        (i-1, j+1),
        (i+1, j-1),
        (i+1, j+1)
    ]

    return [n for n in neighbors if 0 <= n[1] < map.shape[0] and 0 <= n[0] < map.shape[1]]

def get_neighbors_with_cost(map, loc):
    """
    Returns a list of neighbors for a given location in the map with their cost to come

    Parameters:
        map (numpy.ndarray): The image
        loc (tuple): The location as a tuple (x, y)
    
    Returns:
        list: A list of tuples representing the neighbors as ((x, y), cost)
    """
    i, j = loc
    root_2 = np.sqrt(2)
    neighbors = [
        ((i-1, j), 1),
        ((i+1, j), 1),
        ((i, j-1), 1),
        ((i, j+1), 1),
        ((i-1, j-1), root_2),
        ((i-1, j+1), root_2),
        ((i+1, j-1), root_2),
        ((i+1, j+1), root_2)
    ]

    return [n for n in neighbors if 0 <= n[0][1] < map.shape[0] and 0 <= n[0][0] < map.shape[1]]

def get_free_neighbors(map, loc):
    """
    Returns a list of neighbors in free space for a given location in the map

    Parameters:
        map (numpy.ndarray): The image
        loc (tuple): The location as a tuple (x, y)
    
    Returns:
        list: A list of tuples representing the free neighbors as (x, y)
    """
    i, j = loc
    neighbors = [
        (i-1, j),
        (i+1, j),
        (i, j-1),
        (i, j+1),
        (i-1, j-1),
        (i-1, j+1),
        (i+1, j-1),
        (i+1, j+1)
    ]

    return [n for n in neighbors if 0 <= n[1] < map.shape[0] and 0 <= n[0] < map.shape[1] and is_free(map, n)]

def get_free_neighbors_with_cost(map, loc):
    """
    Returns a list of neighbors in free space for a given location in the map
    with their cost to come

    Parameters:
        map (numpy.ndarray): The image
        loc (tuple): The location as a tuple (x, y)
    
    Returns:
        list: A list of tuples representing the neighbors as ((x, y), cost)
    """
    i, j = loc
    root_2 = np.sqrt(2)
    neighbors = [
        ((i-1, j), 1),
        ((i+1, j), 1),
        ((i, j-1), 1),
        ((i, j+1), 1),
        ((i-1, j-1), root_2),
        ((i-1, j+1), root_2),
        ((i+1, j-1), root_2),
        ((i+1, j+1), root_2)
    ]

    return [n for n in neighbors if 0 <= n[0][1] < map.shape[0] and 0 <= n[0][0] < map.shape[1] and is_free(map, n[0])]

def has_free_neighbor(map, loc):
    """
    Returns a boolean indicating if a location in the map has a free neighbor

    Parameters:
        map (numpy.ndarray): The image
        loc (tuple): The location as a tuple (x, y)
    
    Returns:
        bool: True if the location has a free neighbor, False otherwise
    """
    height, width = map.shape
    i, j = loc

    i_min = max(0, i-1)
    i_max = min(height, i+2)
    j_min = max(0, j-1)
    j_max = min(width, j+2)
    
    for x in range(i_min, i_max):
        for y in range(j_min, j_max):
            if (x, y) != loc and map[y, x] == 255:
                return True
    return False


# ----------------------- A* Path Helpers ------------------------
def generate_alternate_goal(visited_points, goal):
    """
    Returns the point in the visited_points list closest to the goal

    Parameters:
        visited_points (numpy.ndarray): (x, y) pairs that were reachable
        goal: (x, y) of the goal that was found to be unreachable

    Returns:
        tuple: (x, y) of the point closest to the goal
    """
    distances = np.linalg.norm(visited_points - goal, axis=1)

    return tuple(visited_points[np.argmin(distances)])

def reconstruct_path(visited, goal, map_metadata):
    """
    Given the visited data structure, a goal location, and map metadata, returns a path
    from the robot's location to the goal.

    Parameters:
        visited (dict): A dictionary of (x, y) pairs mapped to (_, parent_node, _), where
                        the robot's starting location has a parent of None
        goal (tuple): (x, y) pair representing the goal location
        map_metadata (MapMetadata): The current map metadata
    """
    path = []
    current = goal
    while current is not None:
        # Convert point from map space to free space
        current_x_in_space = current[0] * map_metadata.resolution + map_metadata.origin.position.x
        current_y_in_space = current[1] * map_metadata.resolution + map_metadata.origin.position.y

        path.append((current_x_in_space, current_y_in_space))
        current = visited[current][1]

    # We construct the path from the goal to the robot, but need the reverse
    path.reverse()

    return path

def a_star(map, robot_loc, goal, map_metadata):
    """
    Use A* to find the shortest path from the robot's location to the goal. If the goal
    is not adjacent to free space, the goal will be adjusted to the closest point in free
    space to the goal. All parameters and return values are expected to be in the map
    space.

    Parameters:
        map (numpy.ndarray): A thresholded image in the configuration space of the robot
        robot_loc (tuple): The location of the robot as (x, y) coordinates
        goal (tuple): The target location as (x, y) coordinates
        map_metadata (MapMetadata): The current map metadata
    Returns:
        list: A list of tuples representing the path from the robot's location to the
              goal location
    """
    rospy.loginfo("Starting A*")

    # Initialize data structures for A*
    # visited stores (distance from robot, parent node, is node closed) and is indexed using (i,j) tuple
    priority_queue = []
    heapq.heappush(priority_queue, (0, robot_loc))
    visited = {robot_loc: (0, None, False)}

    # While the list is not empty - use a break for if the node is the end node
    while priority_queue:
        _, curr_node = heapq.heappop(priority_queue)
        curr_node_distance, curr_node_parent, curr_node_closed = visited[curr_node]

        # If this node is closed, skip it
        if curr_node_closed:
            continue

        # NOTE: After refactoring new_find_best_point to select known free space as the
        # goal, we can change this for a direct comparison between goal and curr_node
        # If we found the goal, stop
        if goal in get_neighbors(map, curr_node):
            # We don't want to park on the goal, it might not be a safe area
            goal = curr_node
            break

        # Close this node
        visited[curr_node] = (curr_node_distance, curr_node_parent, True)

        # We use get_free_neighbors, so we don't have to check if nodes are obstacles
        for neighbor, neighbor_cost in get_free_neighbors_with_cost(map, curr_node):
                # If a neighbor is closed, skip it
                if visited.get(neighbor, (0, None, False))[1]:
                    continue
                
                # Calculate distance from robot
                distance = curr_node_distance + neighbor_cost

                # If we haven't visited this neighbor or found a shorter route to it, update it
                if neighbor not in visited or distance < visited[neighbor][0]:
                    heuristic = np.linalg.norm((neighbor[0] - goal[0], neighbor[1] - goal[1]))
                    visited[neighbor] = (distance, curr_node, False)
                    heapq.heappush(priority_queue, (distance + heuristic, neighbor))


    # If we can't path to the goal, path as close to the goal as possible
    if not goal in visited:
        old_goal = goal
        visited_points = np.array(list(visited.keys()))

        goal = generate_alternate_goal(visited_points, goal)
        
        save_map_as_debug_image("visited_points", map, visited_points, old_goal, robot_loc)
        rospy.logerr(f"Goal {old_goal} was unreachable; routing to {goal} instead.")

    # Reconstruct the path to the goal using the parent nodes stored in visited
    path = reconstruct_path(visited, goal, map_metadata)

    return path

def multi_goal_a_star(map, robot_loc, goals):
    """
    Use A* to get the distance to a set of points from the robot's location. This
    function is intended to be used for state updation, or when you want to maintain the
    distances of points to the robot's location. All parameters and return values are
    expected to be in the map space. Unreachable goal points will be discarded from the
    return value.

    Parameters:
        map (numpy.ndarray): A thresholded image in the configuration space of the robot
        robot_loc (tuple): The location of the robot as (x, y) coordinates
        goals (set): A set of (x, y) pairs representing target locations

    Returns:
        dict: A mapping of (x, y) pairs to distances from the robot location. Goals that
              require traveling through obstacles to reach will be discarded.
    """
    # Initialize data structures for multi-goal A*
    # visited stores (distance from robot, is node closed)
    # goal_distances maps goals to their distance from the robot
    priority_queue = [(0, robot_loc)]
    visited = {robot_loc: (0, False)}
    remaining_goals = np.array(list(goals))
    goal_distances = {}
    
    while priority_queue and len(remaining_goals):
        _, curr_node = heapq.heappop(priority_queue)
        curr_node_distance, curr_node_closed = visited[curr_node]

        # If this node is closed, skip it
        if curr_node_closed:
            continue

        # Check if this is a goal node we haven't found yet
        if np.any(np.all(remaining_goals == curr_node, axis=1)):
            # Store distance from the robot to the goal
            goal_distances[curr_node] = curr_node_distance

            # Remove curr_node while maintaining shape of remaining_goals
            mask = ~np.all(remaining_goals == curr_node, axis=1)
            remaining_goals = remaining_goals[mask]

            if len(remaining_goals) == 0:
                break

        # Close this node
        visited[curr_node] = (curr_node_distance, True)

        # NOTE: When we change exploring.py to select known free space as goal and we do the thing where we don't process neighbors of rejected goals,
        #       we should be able to change this to not is_free instead because points in unknown space won't make it in the PQ
        # We use get_neighbors because there is a chance for a goal to be in unseen space
        for neighbor, neighbor_cost in get_neighbors_with_cost(map, curr_node):
            # If a neighbor is closed, skip it
            if visited.get(neighbor, (0, False))[1]:
                continue

            # Skip obstacles since we can't pass through them
            if is_wall(map, neighbor):
                continue

            # Calculate distance from robot
            distance = curr_node_distance + neighbor_cost

            # If we haven't visited this neighbor or found a shorter route to it, update it
            if neighbor not in visited or distance < visited[neighbor][0]:
                visited[neighbor] = (distance, False)

                # Use minimum distance to any remaining goal as heuristic
                heuristic = min(np.linalg.norm(remaining_goals - neighbor, axis=1))
                heapq.heappush(priority_queue, (distance + heuristic, neighbor))
    
    return goal_distances
