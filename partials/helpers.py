def world_to_map(x, y, map):
    grid_x = int((x - map.info.origin.position.x) / map.info.resolution)
    grid_y = int((y - map.info.origin.position.y) / map.info.resolution)

    x_out_of_bounds = grid_x < 0 or grid_x >= map.info.width
    y_out_of_bounds = grid_y < 0 or grid_y >= map.info.height
    out_of_bounds = x_out_of_bounds or y_out_of_bounds

    if out_of_bounds:
        return None

    return grid_x + grid_y * map.info.width

def grid_to_world(index, map):
    grid_x = index % map.info.width
    grid_y = index // map.info.width
    
    world_x = grid_x * map.info.resolution + map.info.origin.position.x
    world_y = grid_y * map.info.resolution + map.info.origin.position.y
    
    return (world_x, world_y)
