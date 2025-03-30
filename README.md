# Autonomous Mapping of Unseen Environments

## Project Overview
This project implements an autonomous mapping system for a simulated TurtleBot equipped with a LIDAR sensor in ROS. The robot operates under non-holonomic constraints and must explore an unknown environment to generate a complete map autonomously.

The system leverages the gmapping package for SLAM, focusing on two critical components:

1. **Low-level controller**: Handles obstacle avoidance and robot control
2. **High-level controller**: Determines optimal exploration strategy, selects goal points, and creates waypoints

I am most proud of my expanding wavefront frontier detection algorithm, an algorithm I developed independently and upon completion realized it had been [published by Phillip Quin, et al.](https://opus.lib.uts.edu.au/bitstream/10453/30533/1/quinACRA2014.pdf)

## System Architecture

### Low-Level Controller (`student_driver.py`)
The low-level controller manages the robot's movement to the current waypoint, leveraging a sophisticated obstacle-avoidance system when necessary.

Technical implementation details:
- **Obstacle-free path control**:
  - Implements proportional control with saturation for both linear and angular velocity
  - Uses linear scaling based on distance and angle to target with carefully tuned factors
- **Obstacle avoidance algorithm**:
  - Dynamically calculates the "angle of concern" using `2 * abs(np.arctan(self._robot_maximal_radius / obstacle_distance))`
    - The "angle of concern" is the angle that must read as free space for the robot to pass through safely
  - Employs `np.lib.stride_tricks.sliding_window_view` to efficiently find all potential safe cones
  - Selects optimal safe cone by minimizing required rotation
  - Uses carefully tuned tanh functions to create smooth, bounded control responses

### High-Level Controller (`student_controller.py`)
The high-level controller implements sophisticated frontier detection and selection strategies to determine where the robot should explore next. The system includes multiple approaches with different tradeoffs:

#### Goal Point Selection (Frontier Detection)
**1. Convolutional Frontier Detection**
- Employs a convolution-based approach to identify frontier points
- Time complexity: O(n) where n is the number of cells in the map
- Limitation: May identify unreachable frontiers (e.g., inside walls)

**2. Information-Theoretic Approach (find_highest_information_gain_point)**
- Selects points based on potential information gain
- Generates a circular kernel for a specified radius
- Uses convolution to calculate information density at each point: `information_gain_map = convolve(unseen_points, kernel, mode='constant')`
- Identifies the point with maximum potential gain
- Limitation: Considers points through walls when calculating information gain

**3. Expanding Wavefront Frontier Detection (Primary Algorithm)**
- Implements a complete, efficient O(n) frontier detection algorithm
- Maintains global, persistent data structures (`is_closed`, `priority_queue`) to avoid reprocessing points
- Handles dynamic map updates by recalculating distances for stale values using a multi-goal A* algorithm
- Uses a modified Dijkstra's algorithm to expand outward from the robot
- Implements distance restriction to ensure exploration progress
- Tracks rejected candidate goals for future consideration
- Ensures exploration completeness through careful frontier management

#### Path Planning

Implements A* algorithm for path finding, however the resolution of the path is too high, so waypoints must be generated from the original path.

**Waypoint Generation**
- Implements line-of-sight pruning using Bresenham's algorithm to reduce waypoint count
- The `has_line_of_sight` function checks for obstacles between points
- Converts map coordinates to world coordinates for the robot controller

## Repository Structure

This repository contains multiple labs, with Lab 3 being the primary focus and final implementation of the autonomous mapping system.

```
src/
│
├── lab0/     # Preliminary testing environment (not essential)
├── lab1/     # Component testing environment (not essential)
├── lab2/     # Integration testing environment (not essential)
│
└── lab3/     # Main autonomous mapping implementation
    ├── launch/
    │   └── lab3.launch          # Main launch file for the system
    └── src/
        ├── student_driver.py    # Low-level controller with obstacle avoidance
        │
        ├── student_controller.py # High-level controller for exploration
        │
        ├── exploring.py         # Frontier detection algorithms
        │
        ├── path_planning.py     # A* implementation and map utilities
        │
        └── helpers.py           # Utility functions for mapping and coordinates
```

Labs 0, 1, and 2 were developmental stages used to test individual robot functionalities and are not essential to understanding the final autonomous mapping implementation.


## Installation and Usage

### Prerequisites
- ROS (Robot Operating System) Noetic
- RViz for visualization
- gmapping SLAM package

### Setup
1. Clone this repository into your ROS workspace src directory
2. Build the workspace: `catkin_make`
3. Source the workspace: `source devel/setup.bash`

### Running the System
Launch the autonomous mapping system in the simulated environment:
```
roslaunch lab3 lab3.launch
```

## Future Work
See issues.
