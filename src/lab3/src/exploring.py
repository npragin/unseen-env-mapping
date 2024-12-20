#!/usr/bin/env python3

# This assignment lets you both define a strategy for picking the next point to explore and determine how you
#  want to chop up a full path into way points. You'll need path_planning.py as well (for calculating the paths)
#
# Note that there isn't a "right" answer for either of these. This is (mostly) a light-weight way to check
#  your code for obvious problems before trying it in ROS. It's set up to make it easy to download a map and
#  try some robot starting/ending points
#
# Given to you:
#   Image handling
#   plotting
#   Some structure for keeping/changing waypoints and converting to/from the map to the robot's coordinate space
#
# Slides

# The ever-present numpy
import numpy as np
from scipy.ndimage import convolve

# Your path planning code
import path_planning as path_planning
# Our priority queue
import heapq

# Using imageio to read in the image
import rospy
from helpers import world_to_map


# -------------- Showing start and end and path ---------------
def plot_with_explore_points(im_threshhold, zoom=1.0, robot_loc=None, explore_points=None, best_pt=None):
    """Show the map plus, optionally, the robot location and points marked as ones to explore/use as end-points
    @param im - the image of the SLAM map
    @param im_threshhold - the image of the SLAM map
    @param robot_loc - the location of the robot in pixel coordinates
    @param best_pt - The best explore point (tuple, i,j)
    @param explore_points - the proposed places to explore, as a list"""

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
            if is_reachable(im_thresh, (i, j)):
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


# -------------- For converting to the map and back ---------------
def convert_pix_to_x_y(im_size, pix, size_pix):
    """Convert a pixel location [0..W-1, 0..H-1] to a map location (see slides)
    Note: Checks if pix is valid (in map)
    @param im_size - width, height of image
    @param pix - tuple with i, j in [0..W-1, 0..H-1]
    @param size_pix - size of pixel in meters
    @return x,y """
    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Pixel {pix} not in image, image size {im_size}")

    return [size_pix * pix[i] / im_size[1-i] for i in range(0, 2)]


def convert_x_y_to_pix(im_size, x_y, size_pix):
    """Convert a map location to a pixel location [0..W-1, 0..H-1] in the image/map
    Note: Checks if x_y is valid (in map)
    @param im_size - width, height of image
    @param x_y - tuple with x,y in meters
    @param size_pix - size of pixel in meters
    @return i, j (integers) """
    pix = [int(x_y[i] * im_size[1-i] / size_pix) for i in range(0, 2)]

    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Loc {x_y} not in image, image size {im_size}")
    return pix


def is_reachable(im, pix):
    """ Is the pixel reachable, i.e., has a neighbor that is free?
    Used for
    @param im - the image
    @param pix - the pixel i,j"""

    # Returns True (the pixel is adjacent to a pixel that is free)
    #  False otherwise
    # You can use four or eight connected - eight will return more points
    # YOUR CODE HERE
    #print(pix)
    neighbors = path_planning.get_neighbors(im, pix)
    if not neighbors:
        return False
    else:
        return True


def find_all_possible_goals(im, map_data):
    """ Find all of the places where you have a pixel that is unseen next to a pixel that is free
    It is probably easier to do this, THEN cull it down to some reasonable places to try
    This is because of noise in the map - there may be some isolated pixels
    @param im - thresholded image
    @return dictionary or list or binary image of possible pixels"""

    # YOUR CODE HERE
    
    # Define masks for unseen and free pixels
    unseen_mask = (im == 128)  # Replace with the actual value for "unseen" pixels
    free_mask = (im == 255)  # Replace with the actual value for "free" pixels
    
    # Define a convolution kernel to check for adjacent unseen pixels
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    
    #print(f"unseen_mask shape: {unseen_mask.shape}")
    #print(f"kernel shape: {kernel.shape}")
    #print(f"map shape: {map.shape}")
    #print(f"freemask shape: {free_mask.shape}")

    robot_height_in_pixels = int(0.44 / map_data.resolution * 1.5)
    robot_kernel = np.ones((robot_height_in_pixels, robot_height_in_pixels))
    
    unseen_or_blocked_areas = (im == 0)
    convolve_result = convolve(unseen_or_blocked_areas, robot_kernel, mode='constant', cval=1)
    free_areas = convolve_result == 0

    # Identify unseen neighbors by convolving the unseen_mask
    unseen_neighbors = convolve(unseen_mask.astype(int), kernel, mode="constant", cval=0)
    # Find free pixels that are adjacent to unseen pixels
    valid_points_mask = free_mask & (unseen_neighbors > 0) & free_areas
    # Get coordinates of valid points
    valid_points = np.argwhere(valid_points_mask)
    # Swap axes if needed
    valid_points_swapped = [(y, x) for (x, y) in valid_points] #np.column_stack((valid_points[:,1], valid_points[:0]))
    #print(valid_points_swapped)
    
    # Filter points to include only reachable ones
    reachable_points = [point for point in valid_points_swapped if is_reachable(im, point)]
    #mask = is_reachable(im, valid_points_swapped)
    #reachable_points = valid_points_swapped[mask]
    # Return as set of tuples (row, column format)
    return set(map(tuple, reachable_points))
    

def find_best_point(possible_points, robot_loc):
    """ Pick one of the unseen points to go to
    @param im - thresholded image
    @param possible_points - possible points to chose from
    @param robot_loc - location of the robot (in case you want to factor that in)
    """
    # YOUR CODE HERE
    min_distance = float('inf')
    closest_point = None
    i, j = robot_loc
    for x, y in possible_points:
        distance = np.sqrt((i - x)**2 + (j - y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x, y)

    return closest_point

visited = None
priority_queue = None

def new_find_best_point(map, map_data, robot_loc):
    #TODO: When storing visited and priority_queue in the global scope, the previous distances we rank the nodes by will be stale, how can we fix this?
    rospy.loginfo("Starting new_find_best_point")

    robot_height_in_pixels = int(0.44 / map_data.resolution * 1.5)

    # Convolution to avoid pathing too close to the wall
    kernel = np.ones((robot_height_in_pixels, robot_height_in_pixels))
    # TODO: Figure out if this filters out unseen areas, seems like it only filters out walls
    unseen_or_blocked_areas = (map == 0)
    convolve_result = convolve(unseen_or_blocked_areas, kernel, mode='constant', cval=1)
    free_areas = convolve_result == 0

    # Allow processing points too close to the wall if robot is too close to the wall
    process_bad_nodes = np.sum(free_areas[max(0, robot_loc[1] - 1):min(free_areas.shape[0], robot_loc[1] + 2), max(0, robot_loc[0] - 1):min(free_areas.shape[1], robot_loc[0] + 2)]) < 2
    
    # Initialize data structures for Dijkstra
    # Visited stores (distance from robot, parent node, is node closed)
    global visited, priority_queue
    if visited is None and priority_queue is None:
        priority_queue = []
        heapq.heappush(priority_queue, (0, robot_loc))
        visited = {}
        visited[robot_loc] = (0, None, False)
    
    nearest = None
    while priority_queue:
        curr_node = heapq.heappop(priority_queue)[1]
        curr_node_distance, curr_node_parent, curr_node_closed = visited[curr_node]

        if curr_node_closed:
            continue

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue

                neighbor = (curr_node[0] + di, curr_node[1] + dj)

                # if not is_free(map, neighbor):
                #     continue

                if not free_areas[neighbor[1], neighbor[0]] and not process_bad_nodes:
                    continue

                if process_bad_nodes and free_areas[neighbor[1], neighbor[0]]:
                    rospy.logerr("We were processing bad nodes, but no longer!")
                    process_bad_nodes = False
                
                neighbor_distance = curr_node_distance + np.linalg.norm((di, dj))

                if neighbor not in visited or neighbor_distance < visited[neighbor][0]:
                    visited[neighbor] = (neighbor_distance, curr_node, False)
                    heapq.heappush(priority_queue, (neighbor_distance, neighbor))

        # Close the node
        visited[curr_node] = (curr_node_distance, curr_node_parent, True)

        if map[curr_node[1], curr_node[0]] == 128 and np.linalg.norm((curr_node[1] - robot_loc[1], curr_node[0] - robot_loc[0])) > np.linalg.norm((0.38 / map_data.resolution, 0.44 / map_data.resolution)):
            rospy.logerr(f"BREAKING AND BEST IS {curr_node}")
            nearest = curr_node
            break

    rospy.logerr("We made it out!")
    if len(priority_queue) == 0:
        rospy.logerr("We might be done?")

    # Did this to use with other A* algorithm
    return visited[nearest][1]

    path = []
    current = nearest
    while current:
        current_x_in_space = current[0] * map_data.resolution + map_data.origin.position.x
        current_y_in_space = current[1] * map_data.resolution + map_data.origin.position.y
        path.append((current_x_in_space, current_y_in_space))
        current = visited[current][1]
        
    path.reverse()
    return path

def find_closest_point(possible_points, robot_loc, map_data):
    points = np.array(list(possible_points))
    distances = np.linalg.norm(points - robot_loc, axis=1)
    robot_map_area = np.linalg.norm((0.22 / map_data.resolution, 0.19 / map_data.resolution))
    idx_not_under_robot = np.where(distances >= robot_map_area * 5)[0]

    best_point_idx = idx_not_under_robot[np.argmin(distances[idx_not_under_robot])]
    best_point = points[best_point_idx]
    rospy.logerr(f"Closest point found is: {best_point}")
    return best_point


def find_furthest_point(possible_points, robot_loc):
    """
    Pick the furthest point to go to.
    
    @param possible_points: possible points to choose from
    @param robot_loc: location of the robot (x, y)
    """
    max_distance = -float('inf')  # Start with the smallest possible value
    furthest_point = None
    i, j = robot_loc

    for x, y in possible_points:
        distance = np.sqrt((i - x)**2 + (j - y)**2)  # Calculate Euclidean distance
        if distance > max_distance:  # Compare with max_distance
            max_distance = distance
            furthest_point = (x, y)

    return furthest_point

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


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def calculate_vector(point1, point2):
    """Calculate the vector from point1 to point2."""
    return point2[0] - point1[0], point2[1] - point1[1]

def find_waypoints(im, path):
    """ Place waypoints along the path
    @param im - the thresholded image
    @param path - the initial path
    @ return - a new path"""

    # Again, no right answer here
    # YOUR CODE HERE
    distance_threshold = 5
    waypoints = []
    cumulative_distance = 0
    previous_vector = None

    for i in range(10, len(path), 10):
        waypoints.append(path[i])

    # Add the last point as a waypoint if it's not already included
    if len(waypoints) == 0 or waypoints[-1] != path[-1]:
        waypoints.append(path[-1])

    return waypoints


if __name__ == '__main__':
    # Doing this here because it is a different yaml than JN
    import yaml_1 as yaml

    im, im_thresh = path_planning.open_image("map.pgm")

    robot_start_loc = (1940, 1953)

    all_unseen = find_all_possible_goals(im_thresh)
    best_unseen = find_best_point(im_thresh, all_unseen, robot_loc=robot_start_loc)

    plot_with_explore_points(im_thresh, zoom=0.1, robot_loc=robot_start_loc, explore_points=all_unseen, best_pt=best_unseen)

    path = path_planning.dijkstra(im_thresh, robot_start_loc, best_unseen, 50)
    waypoints = find_waypoints(im_thresh, path)
    path_planning.plot_with_path(im, im_thresh, zoom=0.1, robot_loc=robot_start_loc, goal_loc=best_unseen, path=waypoints)

    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    # plt.show()

    print("Done")
