# Dustbot (ROS 2)

This repository implements a simplified Dustbot in a 2D NxN grid world using ROS 2.
The system is composed of two main nodes: `world` and `robot`.
Communication is organized through 2 topics (pub/sub) and 2 services (client/server), as required by the assignment.

## Project structure

### Packages
- `dustbot_interface`: contains custom message (`.msg`) and service (`.srv`) definitions used by the system.
- `dustbot`: contains the main nodes (`world`, `robot`) that use the interfaces.

## Communication

### Topics
- `/dustbot/garbage_position`  
  Publishes the current garbage position in the grid. The grid is NxN and the size `N` is initialized at startup via a launch file.

- `/dustbot/global_position`  
  Publishes the robot position in the grid. The robot starts at (0, 0).

### Services
- `/dustbot/load_garbage`  
  Used by the `robot` node to request the garbage pick-up (loading).

- `/dustbot/set_direction`  
  Used by the `robot` node to request a change in the robot movement direction.

## Interfaces (`dustbot_interface`)

### Messages
- `GridPosition.msg`  
  Message used to represent a grid position (used by both topics).
  It contains two integers: `x` and `y`.

  Note on movement (direction vectors):
  - N = (0, -1)
  - S = (0, 1)
  - E = (1, 0)
  - W = (-1, 0)

### Services
- `GarbagePickUp.srv`  
  Service interface for `/dustbot/load_garbage`.
  The robot sends a boolean request (e.g., `pick = True`) and receives a boolean response (e.g., `success`) indicating whether the pick-up succeeded.

- `SetDirection.srv`  
  Service interface for `/dustbot/set_direction`.
  - Request: `movement_x`, `movement_y` (integers)
  - Response: `success` (boolean)

## Nodes (`dustbot`)

### `world` node
- Publishes robot and garbage positions on the two topics.
- Implements the two service servers:
  - `/dustbot/set_direction`: validates the requested movement (especially grid bounds) and updates the robot movement direction.
  - `/dustbot/load_garbage`: checks that the robot is exactly on the garbage cell and, if so, confirms the pick-up.
- After a successful pick-up, it generates a new random garbage position.
- After `P` pick-ups, it ends the simulation by publishing an “impossible” garbage position `(-1, -1)` to notify the `robot` node that the task is finished, then shuts itself down by completing a `Future` used in the main with `spin_until_future_complete`.

### `robot` node
- Subscribes to:
  - `/dustbot/global_position` (robot current position)
  - `/dustbot/garbage_position` (current target)
- Computes the direction to follow based on the robot position and the target position.
- Calls:
  - `/dustbot/set_direction` to set/update the movement direction.
  - `/dustbot/load_garbage` when the robot reaches the garbage cell.
- When it receives `(-1, -1)` on `/dustbot/garbage_position`, it interprets it as end of simulation and shuts down(it uses the same method of the world node).

## Parameters
- `N`: grid size (NxN)
- `P`: number of pick-ups to complete before ending the simulation  
Both are loaded at startup via a launch file.

## How to run

### Build
From the root of your ROS 2 workspace:

```bash
colcon build --packages-select dustbot_interface dustbot
source ~/ros2_ws/install/setup.bash
ros2 launch dustbot dustbot_launch.xml

