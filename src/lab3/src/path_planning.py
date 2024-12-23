#!/usr/bin/env python3

# This assignment implements Dijkstra's shortest path on a graph, finding an unvisited node in a graph,
#   picking which one to visit, and taking a path in the map and generating waypoints along that path
#
# Given to you:
#   Priority queue
#   Image handling
#   Eight connected neighbors
#
# Slides https://docs.google.com/presentation/d/1XBPw2B2Bac-LcXH5kYN4hQLLLl_AMIgoowlrmPpTinA/edit?usp=sharing

# The ever-present numpy
import numpy as np

# Our priority queue
import heapq
import rospy
from scipy.ndimage import convolve
from helpers import save_map_image
from exploring import find_highest_concentration_point


# -------------- Showing start and end and path ---------------
def plot_with_path(im, im_threshhold, zoom=1.0, robot_loc=None, goal_loc=None, path=None):
    """Show the map plus, optionally, the robot location and goal location and proposed path
    @param im - the image of the SLAM map
    @param im_threshhold - the image of the SLAM map
    @param zoom - how much to zoom into the map (value between 0 and 1)
    @param robot_loc - the location of the robot in pixel coordinates
    @param goal_loc - the location of the goal in pixel coordinates
    @param path - the proposed path in pixel coordinates"""

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


# -------------- Thresholded image True/False ---------------
def is_wall(im, pix):
    """ Is the pixel a wall pixel?
    @param im - the image
    @param pix - the pixel i,j"""
    if im[pix[1], pix[0]] == 0:
        return True
    return False


def is_unseen(im, pix):
    """ Is the pixel one we've seen?
    @param im - the image
    @param pix - the pixel i,j"""
    if im[pix[1], pix[0]] == 128:
        return True
    return False


#def is_free(im, pix):
    """ Is the pixel empty?
    @param im - the image
    @param pix - the pixel i,j"""
    if im[pix[1], pix[0]] == 255:
        return True
    return False

def is_free(im, pix):
    """
    Checks if the pixel is free.
    @param im: The thresholded image
    @param pix: The pixel coordinate as a tuple (x, y)
    @return: True if the pixel is free (value == 255), False otherwise
    """
    if not isinstance(pix, tuple) or len(pix) != 2:
        raise ValueError(f"Invalid pixel coordinate: {pix}")
    
    # Convert to integers
    pix = (int(pix[0]), int(pix[1]))
    
    # Check bounds
    if not (0 <= pix[1] < im.shape[0] and 0 <= pix[0] < im.shape[1]):
        raise IndexError(f"Pixel {pix} is out of bounds for image shape {im.shape}")
    
    # Check if pixel is free
    if im[pix[1], pix[0]] == 255:
        return True
    return False


def convert_image(im, wall_threshold, free_threshold):
    """ Convert the image to a thresholded image with not seen pixels marked
    @param im - WXHX ?? image (depends on input)
    @param wall_threshold - number between 0 and 1 to indicate wall
    @param free_threshold - number between 0 and 1 to indicate free space
    @return an image of the same WXH but with 0 (free) 255 (wall) 128 (unseen)"""

    # Assume all is unseen
    im_ret = np.zeros((im.shape[0], im.shape[1]), dtype='uint8') + 128

    im_avg = im
    if len(im.shape) == 3:
        # RGB image - convert to gray scale
        im_avg = np.mean(im, axis=2)
    # Force into 0,1
    im_avg = im_avg / np.max(im_avg)
    # threshold
    #   in our example image, black is walls, white is free
    im_ret[im > wall_threshold] = 0
    im_ret[(im < free_threshold) & (im != -1)] = 255
    return im_ret



# -------------- Getting 4 or 8 neighbors ---------------
def four_connected(pix):
    """ Generator function for 4 neighbors
    @param im - the image
    @param pix - the i, j location to iterate around"""
    for i in [-1, 1]:
        ret = pix[0] + i, pix[1]
        yield ret
    for i in [-1, 1]:
        ret = pix[0], pix[1] + i
        yield ret


def eight_connected(pix):
    """ Generator function for 8 neighbors
    @param im - the image
    @param pix - the i, j location to iterate around"""
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                pass
            ret = pix[0] + i, pix[1] + j
            yield ret

def get_free_neighbors(im, loc):
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
    return [n for n in neighbors if 0 <= n[0] < im.shape[0] and 0 <= n[1] < im.shape[1] and is_free(im, n)]



def dijkstra(im, robot_loc, goal_loc, map_data):
    """ Occupancy grid image, with robot and goal loc as pixels
    @param im - the thresholded image - use is_free(i, j) to determine if in reachable node
    @param robot_loc - where the robot is (tuple, i,j)
    @param goal_loc - where to go to (tuple, i,j)
    @returns a list of tuples"""

    rospy.loginfo("Starting dijkstras")

    robot_height_in_meters = 0.44
    robot_height_in_pixels = robot_height_in_meters / map_data.resolution
    robot_height_in_pixels_with_buffer = int(robot_height_in_pixels * 1.9)

    kernel = np.ones((robot_height_in_pixels_with_buffer, robot_height_in_pixels_with_buffer))
    
    unseen_or_blocked_areas = (im == 0)
    convolve_result = convolve(unseen_or_blocked_areas, kernel, mode='constant', cval=1)
    free_areas = convolve_result == 0

    goal_loc = (goal_loc[0], goal_loc[1])
    process_bad_nodes = np.sum(free_areas[max(0, robot_loc[1] - 1):min(free_areas.shape[0], robot_loc[1] + 2), max(0, robot_loc[0] - 1):min(free_areas.shape[1], robot_loc[0] + 2)]) < 2

    # Initialize data structures for Dijkstra
    # Visited stores (distance from robot, parent node, is node closed) and is indexed using (i,j) tuple
    priority_queue = []
    heapq.heappush(priority_queue, (0, robot_loc))
    visited = {}
    visited[robot_loc] = (0, None, False)

    # While the list is not empty - use a break for if the node is the end node
    while priority_queue:
        current_node = heapq.heappop(priority_queue)
        # Pop returns the value and the i, j
        node_score = current_node[0]
        node_ij = current_node[1]

        # Showing how to get this data back out of visited
        visited_triplet = visited[node_ij]
        visited_distance = visited_triplet[0]
        visited_parent = visited_triplet[1]
        visited_closed_yn = visited_triplet[2]

        #  Step 1: Break out of the loop if node_ij is the goal node
        if node_ij == goal_loc:
            break

        #  Step 2: If this node is closed, skip it
        if visited_closed_yn:
            continue

        #  Step 3: Set the node to closed
        visited[node_ij] = (visited_distance, visited_parent, True)

        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                # Don't do anything for the case where we don't move
                if di == 0 and dj == 0:
                    continue
                
                neighbor = (node_ij[0] + di, node_ij[1] + dj)
                
                # Check if neighbor in direction (di, dj) is valid and free
                if not is_free(im, (neighbor)):
                    continue
                    
                if process_bad_nodes and free_areas[neighbor[1], neighbor[0]]:
                    rospy.logerr("We were processing bad nodes, but no longer!")
                    process_bad_nodes = False

                if not free_areas[neighbor[1], neighbor[0]] and not process_bad_nodes:
                    continue

                # Calculate distance to neighbor and distance to goal and add them to existing cost
                distance = np.linalg.norm((di, dj)) + visited_distance
                heuristic = np.linalg.norm((neighbor[0] - goal_loc[0], neighbor[1] - goal_loc[1]))
                
                # If we haven't tried this path add it to the queue
                if neighbor not in visited or distance < visited[neighbor][0]:
                    visited[neighbor] = (distance, node_ij, False)
                    heapq.heappush(priority_queue, (distance + heuristic, neighbor))

    # If we couldn't path to the goal, path to the closest point to the goal we can
    if not goal_loc in visited:
        old_goal_loc = goal_loc
        visited_points = np.array(list(visited.keys()))
        distances = np.linalg.norm(visited_points - goal_loc, axis=1)
        goal_loc = (visited_points[np.argmin(distances)][0], visited_points[np.argmin(distances)][1])
        rospy.logerr(f"Goal {old_goal_loc} was unreachable, sending {goal_loc} instead.")
        save_map_image("visited_points", im, visited_points, old_goal_loc, robot_loc)
        
    path = []
    current = tuple(goal_loc)
    # Reconstruct the path using the parent nodes stored in the visited dictionary
    while current is not None:
        current_x_in_space = current[0] * map_data.resolution + map_data.origin.position.x
        current_y_in_space = current[1] * map_data.resolution + map_data.origin.position.y
        path.insert(0, (current_x_in_space, current_y_in_space))
        current = visited[current][1]

    return path

def open_image(im_name):
    """ A helper function to open up the image and the yaml file and threshold
    @param im_name - name of image in Data directory
    @returns image anbd thresholded image"""

    # Needed for reading in map info
    from os import open

    im = imageio.imread("Data/" + im_name)

    wall_threshold = 0.7
    free_threshold = 0.9
    try:
        yaml_name = "Data/" + im_name[0:-3] + "yaml"
        with open(yaml_name, "r") as f:
            dict = yaml.load_all(f)
            wall_threshold = dict["occupied_thresh"]
            free_threshold = dict["free_thresh"]
    except:
        pass

    im_thresh = convert_image(im, wall_threshold, free_threshold)
    return im, im_thresh



if __name__ == '__main__':
    # Putting this here because in JN it's yaml
    import yaml_1 as yaml

    # Use one of these

    """ Values for SLAM map
    im, im_thresh = open_image("SLAM_map.png")
    robot_start_loc = (200, 150)
    # Closer one to try
    # robot_goal_loc = (315, 250)
    robot_goal_loc = (615, 850)
    zoom = 0.8
    """

    """ Values for map.pgm"""
    im, im_thresh = open_image("map.pgm")
    robot_start_loc = (1940, 1953)
    robot_goal_loc = (2135, 2045)
    zoom = 0.1

    """
    print(f"Image shape {im_thresh.shape}")
    for i in range(0, im_thresh.shape[1]-1):
        for j in range(0, im_thresh.shape[0]-1):
            if is_free(im_thresh, (i, j)):
                print(f"Free {i} {j}")
    """
    path = dijkstra(im_thresh, robot_start_loc, robot_goal_loc)
    plot_with_path(im, im_thresh, zoom=zoom, robot_loc=robot_start_loc, goal_loc=robot_goal_loc, path=path)

    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    # plt.show()

    print("Done")
