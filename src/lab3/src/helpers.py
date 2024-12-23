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

    # Plot visited points in blue
    ax.scatter(points[:, 0] - 1800, points[:, 1] - 1800, 
            color='blue', marker='.', s=0.25, alpha=1)

    # Plot robot location with yellow star
    ax.scatter([yellow_star[0] - 1800], [yellow_star[1] - 1800], 
            color='yellow', marker='*', s=100)
    
            # Plot goal location with green star
    ax.scatter([green_star[0] - 1800], [green_star[1] - 1800], 
            color='green', marker='*', s=5)
    
    ax.invert_yaxis()
    plt.colorbar()
    plt.savefig(os.path.expanduser(f"~/ros_ws/src/lab3/images/{filename}.png"))
    rospy.logerr(f"Saved visited points visualization as {filename}")