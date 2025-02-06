#!/usr/bin/env python3

import rospy
import numpy as np
import heapq
import cv2

from helpers import save_map_as_debug_image


# ------------------ Plotting path, robot, and goal location ------------------
def plot_with_path(im, im_threshhold, zoom=1.0, robot_loc=None, goal_loc=None, path=None):
    """
    Plot the map plus, optionally, the robot location and goal location and proposed path

    Parameters:
        im (numpy.ndarray): The original image
        im_threshhold (numpy.ndarray): The thresholded image
        zoom (float): The zoom level
        robot_loc (tuple): The robot location as a tuple (x, y)
        goal_loc (tuple): The goal location as a tuple (x, y)
        path (list): A list of tuples representing the path
    """

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im, origin='lower', cmap="gist_gray")
    axs[0].set_title("original image")
    axs[1].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[1].set_title("threshold image")
    """
    # Used to double check that the is_xxx routines work correctly
    for i in range(0, im_threshhold.shape[1]-1, 10):
        for j in range(0, im_threshhold.shape[0]-1, 10):
            if is_wall(im_thresh, (i, j)):
                axs[1].plot(i, j, '.b')
    """

    # Double checking lower left corner
    axs[1].plot(10, 5, 'xy', markersize=5)

    # Show original and thresholded image
    for i in range(0, 2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if goal_loc is not None:
            axs[i].plot(goal_loc[0], goal_loc[1], '*g', markersize=10)
        if path is not None:
            for p, q in zip(path[0:-1], path[1:]):
                axs[i].plot([p[0], q[0]], [p[1], q[1]], '-y', markersize=2)
                axs[i].plot(p[0], p[1], '.y', markersize=2)
        axs[i].axis('equal')

    for i in range(0, 2):
        # Implements a zoom - set zoom to 1.0 if no zoom
        width = im.shape[1]
        height = im.shape[0]

        axs[i].set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        axs[i].set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)


# -------------- Thresholded image pixel identification ---------------
def is_wall(im, pix):
    """
    Returns true if the pixel in the image is an obstacle

    Parameters:
        im (numpy.ndarray): The thresholded image
        pix (tuple): The pixel coordinate as a tuple (x, y)
    """
    if im[pix[1], pix[0]] == 0:
        return True
    return False

def is_unseen(im, pix):
    """
    Returns true if the pixel in the image is unseen

    Parameters:
        im (numpy.ndarray): The thresholded image
        pix (tuple): The pixel coordinate as a tuple (x, y)
    """
    if im[pix[1], pix[0]] == 128:
        return True
    return False

def is_free(im, pix):
    """
    Returns true if the pixel in the image is free space

    Parameters:
        im (numpy.ndarray): The thresholded image
        pix (tuple): The pixel coordinate as a tuple (x, y)
    """
    if im[pix[1], pix[0]] == 255:
        return True
    return False


# -------------- Occupancy grid to threshold image conversion ---------------
def convert_image(im, wall_threshold, free_threshold):
    """
    Convert the OccupancyGrid to a thresholded image. Any points that have a confidence
    level below what is required by the provided thresholds will be marked as unseen
    space. The threshold image will have 3 values:
        - 0 representing an obstacle
        - 128 representing unseen space
        - 255 representing free space

    Parameters:
        im (numpy.ndarray): An OccupancyGrid's data member resized to a 2D array
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
                        The dimensions will match the im parameter
    """
    # Assume all is unseen
    im_ret = np.zeros((im.shape[0], im.shape[1]), dtype='uint8') + 128

    # Set cells with confidence above threshold values to free or obstacle
    im_ret[im > wall_threshold] = 0
    im_ret[(im < free_threshold) & (im != -1)] = 255
    return im_ret

def convert_map_to_configuration_space(im, wall_threshold, free_threshold, robot_dimension):
    """
    Convert the OccupancyGrid to a thresholded image in the configuration space.
    Any points that have a confidence level below what is required by the provided
    thresholds will be marked as unseen space. The threshold image will have 3 values:
        - 0 represents an obstacle
        - 128 represents unseen space
        - 255 represents free space

    Parameters:
        im (numpy.ndarray): An OccupancyGrid's data member resized to a 2D array
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
                        The dimensions will match the im parameter
    """

    # Assume all is unseen
    im_ret = np.zeros((im.shape[0], im.shape[1]), dtype='uint8') + 128

    # Set cells with confidence above threshold values to free
    im_ret[(im < free_threshold) & (im != -1)] = 255

    # Use mask indicating walls to dilate the obstacles by robot_dimension / 2
    wall_mask = (im > wall_threshold).astype(np.uint8)
    kernel = np.ones((robot_dimension, robot_dimension), dtype='uint8')
    dilated = cv2.dilate(wall_mask, kernel, iterations=1)
    im_ret[dilated > 0] = 0
    
    return im_ret


# ----------------------- Getting neighbors -----------------------
def get_neighbors(im, loc):
    """
    Returns a list of neighbors for a given location in the image

    Parameters:
        im (numpy.ndarray): The image
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

    return [n for n in neighbors if 0 <= n[1] < im.shape[0] and 0 <= n[0] < im.shape[1]]

def get_neighbors_with_cost(im, loc):
    """
    Returns a list of neighbors for a given location in the image with their cost to come

    Parameters:
        im (numpy.ndarray): The image
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

    return [n for n in neighbors if 0 <= n[0][1] < im.shape[0] and 0 <= n[0][0] < im.shape[1]]

def get_free_neighbors(im, loc):
    """
    Returns a list of neighbors in free space for a given location in the image

    Parameters:
        im (numpy.ndarray): The image
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

    return [n for n in neighbors if 0 <= n[1] < im.shape[0] and 0 <= n[0] < im.shape[1] and is_free(im, n)]

def get_free_neighbors_with_cost(im, loc):
    """
    Returns a list of neighbors in free space for a given location in the image
    with their cost to come

    Parameters:
        im (numpy.ndarray): The image
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

    return [n for n in neighbors if 0 <= n[0][1] < im.shape[0] and 0 <= n[0][0] < im.shape[1] and is_free(im, n[0])]

def has_free_neighbor(im, loc):
    """
    Returns a boolean indicating if a location in the image has a free neighbor

    Parameters:
        im (numpy.ndarray): The image
        loc (tuple): The location as a tuple (x, y)
    
    Returns:
        bool: True if the location has a free neighbor, False otherwise
    """
    height, width = im.shape
    i, j = loc

    i_min = max(0, i-1)
    i_max = min(height, i+2)
    j_min = max(0, j-1)
    j_max = min(width, j+2)
    
    for x in range(i_min, i_max):
        for y in range(j_min, j_max):
            if (x, y) != loc and im[y, x] == 255:
                return True
    return False


# ----------------------- A* Path Helpers ------------------------
def generate_alternate_goal(visited_points, goal_loc):
    """
    Returns the point in the visited_points list closest to the goal_loc.

    Parameters:
        visited_points (list of tuples): (x, y) pairs that were reachable
        goal_loc: (x, y) of the goal that was found to be unreachable

    Returns:
        tuple: (x, y) of the point closest to the goal_loc
    """
    visited_points = np.array(visited_points)
    distances = np.linalg.norm(visited_points - goal_loc, axis=1)

    return tuple(visited_points[np.argmin(distances)])

def reconstruct_path(visited, goal_loc, map_data):
    """
    Given the visited data structure, a goal location, and map metadata, returns a path
    from the robot's location to the goal.

    Parameters:
        visited (dict): A dictionary of (x, y) pairs mapped to (_, parent_node, _), where
                        the robot's starting location has a parent of None
        goal_loc (tuple): (x, y) pair representing the goal location
        map_data (MapMetadata): The current map metadata
    """
    path = []
    current = goal_loc
    while current is not None:
        # Convert point from map space to free space
        current_x_in_space = current[0] * map_data.resolution + map_data.origin.position.x
        current_y_in_space = current[1] * map_data.resolution + map_data.origin.position.y

        path.append((current_x_in_space, current_y_in_space))
        current = visited[current][1]

    # We construct the path from the goal to the robot, but need the reverse
    path.reverse()

    return path

def a_star(im, robot_loc, goal_loc, map_data):
    """
    Use A* to find the shortest path from the robot's location to the goal location.
    If the goal location is not adjacent to free space, the goal location will be
    adjusted to the closest point in free space to the goal location. All parameters
    and return values are expected to be in the map space.

    Parameters:
        im (numpy.ndarray): A thresholded image in the configuration space of the robot
        robot_loc (tuple): The location of the robot as (x, y) coordinates
        goal_loc (tuple): The target location as (x, y) coordinates
        map_data (MapMetadata): The current map metadata
    Returns:
        list: A list of tuples representing the path from the robot's location to the
              goal location
    """
    rospy.loginfo("Starting A*")

    # Initialize data structures for A*
    # Visited stores (distance from robot, parent node, is node closed) and is indexed using (i,j) tuple
    priority_queue = []
    heapq.heappush(priority_queue, (0, robot_loc))
    visited = {robot_loc: (0, None, False)}

    # While the list is not empty - use a break for if the node is the end node
    while priority_queue:
        curr_node = heapq.heappop(priority_queue)[1]

        # Showing how to get this data back out of visited
        curr_node_distance, curr_node_parent, curr_node_closed = visited[curr_node]

        # If this node is closed, skip it
        if curr_node_closed:
            continue

        # If we found the goal, stop
        if goal_loc in get_neighbors(im, curr_node):
            # We don't want to park on the goal, it might not be a safe area
            goal_loc = curr_node
            break

        # Close this node
        visited[curr_node] = (curr_node_distance, curr_node_parent, True)

        # We use get_free_neighbors, so we don't have to check if nodes are obstacles
        for neighbor, neighbor_cost in get_free_neighbors_with_cost(im, curr_node):
                # Calculate distance from robot and estimated distance to goal
                distance = curr_node_distance + neighbor_cost
                heuristic = np.linalg.norm((neighbor[0] - goal_loc[0], neighbor[1] - goal_loc[1]))

                # If we haven't tried this path add it to the queue
                if neighbor not in visited or distance < visited[neighbor][0]:
                    visited[neighbor] = (distance, curr_node, False)
                    heapq.heappush(priority_queue, (distance + heuristic, neighbor))


    # If we can't path to the goal, path as close to the goal as possible
    if not goal_loc in visited:
        old_goal_loc = goal_loc
        visited_points = list(visited.keys())

        goal_loc = generate_alternate_goal(visited_points, goal_loc)
        
        save_map_as_debug_image("visited_points", im, visited_points, old_goal_loc, robot_loc)
        rospy.logerr(f"Goal {old_goal_loc} was unreachable; routing to {goal_loc} instead.")

    # Reconstruct the path to the goal using the parent nodes stored in visited
    path = reconstruct_path(visited, goal_loc, map_data)

    return path

def old_multi_goal_astar(im, robot_loc, goals):
    """ Occupancy grid image, with robot and goal loc as pixels
    @param im - the thresholded image
    @param robot_loc - where the robot is (tuple, i,j)
    @param goals - list of goals (tuple, i,j)
    @returns a dictionary of distances to each goal"""
    
    free_areas = im != 0
    process_bad_nodes = np.sum(free_areas[max(0, robot_loc[1] - 1):min(free_areas.shape[0], robot_loc[1] + 2), max(0, robot_loc[0] - 1):min(free_areas.shape[1], robot_loc[0] + 2)]) < 2


    open_set = [(0, robot_loc)]  # (f_score, node)
    closed_set = set()
    came_from = {goal: {} for goal in goals}
    g_score = {robot_loc: 0}
    path_distances = {}
    remaining_goals = set(goals)
    
    
    while open_set and remaining_goals:
        current_f, current = heapq.heappop(open_set)

        # Check if this is a goal node we haven't found yet
        if current in remaining_goals:
            path_distances[current] = g_score[current]
        
            remaining_goals.remove(current)

            if len(remaining_goals) == 0:
                break

        if current in closed_set:
            continue

        closed_set.add(current)

        for neighbor, cost in get_neighbors_with_cost(im, current):
            if neighbor in closed_set:
                continue

            if (not free_areas[neighbor[1], neighbor[0]] and not process_bad_nodes) or not has_free_neighbor(im, neighbor):
                continue

            if process_bad_nodes and free_areas[neighbor[1], neighbor[0]]:
                process_bad_nodes = False

            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                # Found a better path to this neighbor
                for goal in remaining_goals:
                    came_from[goal][neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Use minimum f_score to any remaining goal
                min_heuristic = min(np.linalg.norm((neighbor[0] - goal[0], neighbor[1] - goal[1])) 
                          for goal in remaining_goals)
                f_score = tentative_g + min_heuristic
                heapq.heappush(open_set, (f_score, neighbor))
    
    return path_distances

def multi_goal_astar(im, robot_loc, goals):
    """ Occupancy grid image, with robot and goal loc as pixels
    @param im - the thresholded image
    @param robot_loc - where the robot is (tuple, i,j)
    @param goals - list of goals (tuple, i,j)
    @returns a dictionary of distances to each goal"""

    priority_queue = [(0, robot_loc)]
    visited = {robot_loc: (0, False)}
    remaining_goals = np.array(goals)
    goal_distances = {}
    
    while priority_queue and remaining_goals:
        _, curr_node = heapq.heappop(priority_queue)
        curr_node_distance, curr_node_closed = visited[curr_node]

        if curr_node_closed:
            continue

        # Check if this is a goal node we haven't found yet
        if curr_node in remaining_goals:
            goal_distances[curr_node] = curr_node_distance
            remaining_goals.remove(curr_node)

            if len(remaining_goals) == 0:
                break

        visited[curr_node] = (curr_node_distance, True)

        for neighbor, neighbor_cost in get_neighbors_with_cost(im, curr_node):
            # If neighbor is closed, skip it
            # TODO: Test performance with and without this clause to determine whether or not we should keep this
            # If keeping it has better performance consider adding it to A* after performing the test agian
            if visited.get(neighbor, (0, False))[1]:
                continue

            if is_wall(im, neighbor):
                continue

            distance = curr_node_distance + neighbor_cost

            if neighbor not in visited or distance < visited[neighbor][0]:
                # Found a better path to this neighbor, update distance
                visited[neighbor] = (distance, False)
                
                # Use minimum distance to any remaining goal as heuristic
                heuristic = min(np.linalg.norm(remaining_goals - neighbor, axis=1))
        
                heapq.heappush(priority_queue, (distance + heuristic, neighbor))
    
    return goal_distances
