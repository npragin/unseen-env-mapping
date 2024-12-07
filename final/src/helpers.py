def world_to_map(x, y, map_data):
    grid_x = int((x - map_data.origin.position.x) / map_data.resolution)
    grid_y = int((y - map_data.origin.position.y) / map_data.resolution)

    x_out_of_bounds = grid_x < 0 or grid_x >= map_data.width
    y_out_of_bounds = grid_y < 0 or grid_y >= map_data.height
    out_of_bounds = x_out_of_bounds or y_out_of_bounds
 
    if out_of_bounds:
        return None

    return (grid_y, grid_x)

def map_to_world(index, map_data):
    grid_x = index % map_data.width
    grid_y = index // map_data.width

    world_x = grid_x * map_data.resolution + map_data.origin.position.x
    world_y = grid_y * map_data.resolution + map_data.origin.position.y

    return (world_x, world_y)