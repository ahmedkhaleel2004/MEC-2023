from ultralytics import YOLO
import numpy as np
import pygame
import sys

class Car:
    def __init__(self, length, width, initial_x, initial_y):
        # car dimensions in grid cells
        self.length = length
        self.width = width
        
        # initial position in grid cells (top-left corner)
        self.x = initial_x
        self.y = initial_y

    def draw(self, screen):
        # Draw the car as a rectangle on the screen
        car_rect = pygame.Rect(self.x * CELL_SIZE, self.y * CELL_SIZE, self.width * CELL_SIZE, self.length * CELL_SIZE)
        pygame.draw.rect(screen, RED, car_rect)

    def will_collide(self, direction, boxes_array): #pygame collision detection
        # create a copy of the current car rectangle
        future_rect = pygame.Rect(self.x, self.y, self.width, self.length)
        
        # adjust the position of the rectangle based on the direction of movement
        if direction == 'up':
            future_rect.y -= 1
        elif direction == 'down':
            future_rect.y += 1
        elif direction == 'left':
            future_rect.x -= 1
        elif direction == 'right':
            future_rect.x += 1
            
        # check for collision with each bounding box
        for box in boxes_array:
            # create a rectangle for the bounding box
            box_rect = pygame.Rect(int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))# Check if the future car rectangle intersects with this box
            if future_rect.colliderect(box_rect):
                return True  # collision detected
                
        
        return False  # no collision detected

# Initialize the model
model = YOLO('yolov8n.pt')

# Perform detection
results = model(source='parkinglot.jpg', conf=0.2, save=True)

if isinstance(results, list):
    detected_boxes = results[0].boxes

    if not isinstance(detected_boxes.xyxy, np.ndarray):
        boxes_array = detected_boxes.xyxy.cpu().numpy()
    else:
        boxes_array = detected_boxes.xyxy

    
    # 'boxes_array' contains the bounding box information in [x_min, y_min, x_max, y_max, confidence, class] format
else:
    pass

for i in range(len(boxes_array)): #number of detections
    for j in range(4):
        boxes_array[i][j] = int(round(boxes_array[i][j], 1) / 30)

print(boxes_array)

pygame.init()

# Constants
GRID_WIDTH = 50   # Grid width in cells
GRID_HEIGHT = 30  # Grid height in cells
CELL_SIZE = 20    # Size of each cell in pixels
WINDOW_SIZE = (GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE)  # Window size
FPS = 10          # Frames per second

# Colors
GREY = (100, 100, 100) 
RED = (255, 0, 0)

# Setup the screen
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("50x30 Grid Game")
clock = pygame.time.Clock()

# Player settings
user_car = Car(length=4, width=2, initial_x=40, initial_y=15)  # Adjusted starting position

def draw_grid():
    for x in range(0, WINDOW_SIZE[0], CELL_SIZE):
        pygame.draw.line(screen, GREY, (x, 0), (x, WINDOW_SIZE[1]))
    for y in range(0, WINDOW_SIZE[1], CELL_SIZE):
        pygame.draw.line(screen, GREY, (0, y), (WINDOW_SIZE[0], y))

def draw_player():
    rect = pygame.Rect(user_car.length * CELL_SIZE, user_car.width * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, RED, rect)

# Function to draw the detected boxes
def draw_detected_boxes():
    for box in boxes_array:
        # Convert box coordinates to pygame rect format (x, y, width, height)
        x, y, w, h = box[0] * CELL_SIZE, box[1] * CELL_SIZE, (box[2] - box[0]) * CELL_SIZE, (box[3] - box[1]) * CELL_SIZE
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, RED, rect, 1)  # Change '1' to 0 for filled rectangles


# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Handle movement
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and user_car.y > 0 and not user_car.will_collide('up', boxes_array):
        user_car.y -= 1
    if keys[pygame.K_s] and user_car.y + user_car.length < GRID_HEIGHT and not user_car.will_collide('down', boxes_array):
        user_car.y += 1
    if keys[pygame.K_a] and user_car.x > 0 and not user_car.will_collide('left', boxes_array):
        user_car.x -= 1
    if keys[pygame.K_d] and user_car.x + user_car.width < GRID_WIDTH and not user_car.will_collide('right', boxes_array):
        user_car.x += 1

    # Drawing
    screen.fill((0, 0, 0))  # Clear screen
    draw_grid()
    user_car.draw(screen)
    draw_detected_boxes()

    pygame.display.flip()  # Update screen
    clock.tick(FPS)

pygame.quit()
sys.exit()






