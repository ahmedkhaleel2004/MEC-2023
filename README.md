# Parking Spot Finder with YOLOv8 and A* Pathfinding

## Overview
My team and I's submission for the McMaster Engineering Competition in 2023 combines the power of YOLOv8's realtime object detection and the A* pathfinding algorithm to direct users to their preferred parking spot. It scans a top-down view of a parking lot from cameras or sensors, identifies available parking spots, and computes the shortest path to a selected spot.

<div align="center">
  <table>
    <tr>
      <td style="text-align: center; vertical-align: middle; border: 0;">
        <img src="https://github.com/ahmedkhaleel2004/MEC-2023/assets/111161052/51611bec-3ed5-4eff-9bc6-5606bfc47f1d" alt="Alt text for first image" width="400"/>
      </td>
      <td style="text-align: center; vertical-align: middle; border: 0;">
        <img src="https://github.com/ahmedkhaleel2004/MEC-2023/assets/111161052/198d43b8-f6ae-459b-a81d-66bc6a2f1a2e" alt="Alt text for second image" width="400"/>
      </td>
    </tr>
  </table>
</div>


## Key Features
- **Image Scanning**: Uses YOLOv8 for accurate and quick parking spot detection in an image (or video).
- **Efficient Pathfinding**: Implements the A* algorithm for finding the shortest path to the selected parking spot.
- **Interactive**: Users can select an initial and final node with custom locations to visualize different scenarios, along with adding barrier blocks to simulate pedestrians/objects.

## How It Works
1. **Input Data**: Process a frame of a parking lot with object detection and calculate boundary box corners.
2. **Initialization**: Round pixel values into integers and scale according to a discretely initialized grid.
3. **Calculation**: Given an entry and destination node, run A* using grid colours for propogation.
   
## Installation
To run this program, you need to install the following Python libraries:
- Pygame
- Ultralytics YOLO package
- NumPy

You can install these packages using pip:
```bash
pip install pygame ultralytics numpy
