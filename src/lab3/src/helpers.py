def world_to_map(x, y, map_data):
    grid_x = int((x - map_data.origin.position.x) / map_data.resolution)
    grid_y = int((y - map_data.origin.position.y) / map_data.resolution)

    x_out_of_bounds = grid_x < 0 or grid_x >= map_data.width
    y_out_of_bounds = grid_y < 0 or grid_y >= map_data.height
    out_of_bounds = x_out_of_bounds or y_out_of_bounds

    if out_of_bounds:
        return None

    return (grid_x, grid_y)

def map_to_world(index, map_data):
    grid_x = index % map_data.width
    grid_y = index // map_data.width

    world_x = grid_x * map_data.resolution + map_data.origin.position.x
    world_y = grid_y * map_data.resolution + map_data.origin.position.y

    return (world_x, world_y)

def save_map_image(filename, map, points, green_star=None, yellow_star=None):
    import matplotlib.pyplot as plt
    import rospy
    import os

    rospy.loginfo("Saving visited points visualization")

    _, ax = plt.subplots()

    # Plot the base image/map
    plt.imshow(map[1800:2200, 1800:2200], cmap='plasma')

    ax.scatter(points[:, 0] - 1800, points[:, 1] - 1800, color='blue', marker='.', s=0.25, alpha=1)

    if yellow_star:
        ax.scatter([yellow_star[0] - 1800], [yellow_star[1] - 1800], color='yellow', marker='*', s=100)
        
    if green_star:
        ax.scatter([green_star[0] - 1800], [green_star[1] - 1800], color='green', marker='*', s=5)

    ax.invert_yaxis()
    plt.colorbar()
    plt.savefig(os.path.expanduser(f"~/ros_ws/src/lab3/images/{filename}.png"))
    rospy.logerr(f"Saved visited points visualization as {filename}")

def save_map_as_image(map):
    from path_planning import convert_image
    import cv2
    import os

    im = convert_image(map, 0.8, 0.2)

    im = _trim_map_image(im)

    cv2.imwrite(os.path.expanduser("~/ros_ws/src/lab3/images/completed_map.png"), im)

def _trim_map_image(im):
    import numpy as np
    
    top_unseen = 0
    i = 0
    while i < im.shape[0] and np.all(im[i] == 128):
        top_unseen += 1
        i += 1

    bottom_unseen = 0
    i = im.shape[0] - 1
    while i >= 0 and np.all(im[i] == 128):
        bottom_unseen += 1
        i -= 1

        left_unseen = 0
    
    left_unseen = 0
    i = 0
    while i < im.shape[1] and np.all(im[:, i] == 128):
        left_unseen += 1
        i += 1
        
    right_unseen = 0
    i = im.shape[1] - 1
    while i >= 0 and np.all(im[:, i] == 128):
        right_unseen += 1
        i -= 1

    start_row = max(0, top_unseen - 3)
    end_row = min(im.shape[0], im.shape[0] - bottom_unseen + 3)
    start_col = max(0, left_unseen - 3)
    end_col = min(im.shape[1], im.shape[1] - right_unseen + 3)

    im = im[start_row:end_row, start_col:end_col]

    return im