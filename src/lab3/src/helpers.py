def world_to_map(x, y, map_metadata):
    grid_x = int((x - map_metadata.origin.position.x) / map_metadata.resolution)
    grid_y = int((y - map_metadata.origin.position.y) / map_metadata.resolution)

    x_out_of_bounds = grid_x < 0 or grid_x >= map_metadata.width
    y_out_of_bounds = grid_y < 0 or grid_y >= map_metadata.height
    out_of_bounds = x_out_of_bounds or y_out_of_bounds

    if out_of_bounds:
        return None

    return (grid_x, grid_y)

def map_to_world(index, map_metadata):
    grid_x = index % map_metadata.width
    grid_y = index // map_metadata.width

    world_x = grid_x * map_metadata.resolution + map_metadata.origin.position.x
    world_y = grid_y * map_metadata.resolution + map_metadata.origin.position.y

    return (world_x, world_y)

def save_map_as_debug_image(filename, map, points, green_star=None, yellow_star=None):
    import matplotlib.pyplot as plt
    import rospy
    import os

    _, ax = plt.subplots()

    # Plot the base image/map
    plt.imshow(map[1800:2200, 1800:2200], cmap='plasma')

    if points is not None and len(points) > 0:
        ax.scatter(points[:, 0] - 1800, points[:, 1] - 1800, color='cyan', marker='.', s=1, alpha=1)

    if yellow_star:
        ax.scatter([yellow_star[0] - 1800], [yellow_star[1] - 1800], color='black', marker='*', s=25)
        
    if green_star:
        ax.scatter([green_star[0] - 1800], [green_star[1] - 1800], color='green', marker='*', s=25)

    ax.invert_yaxis()
    plt.colorbar()
    plt.savefig(os.path.expanduser(f"~/ros_ws/src/lab3/images/{filename}.png"))
    rospy.logerr(f"Saved debug image as {filename}")

def save_map_as_image(map):
    from path_planning import convert_image
    import cv2
    import os

    map_image = convert_image(map, 0.8, 0.2)

    trimmed_map_image = _trim_map_image(map_image)

    return cv2.imwrite(os.path.expanduser("~/ros_ws/src/lab3/images/completed_map.png"), trimmed_map_image)

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
