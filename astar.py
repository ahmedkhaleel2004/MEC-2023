import pygame
import math
from queue import PriorityQueue
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

results = model(source='parkinglot.jpg', conf=0.2, save=True)

if isinstance(results, list):
    detected_boxes = results[0].boxes

    if not isinstance(detected_boxes.xyxy, np.ndarray):
        boxes_array = detected_boxes.xyxy.cpu().numpy()
    else:
        boxes_array = detected_boxes.xyxy

    
else:
    pass

for i in range(len(boxes_array)): #number of detections
    for j in range(4):
        boxes_array[i][j] = int(round(boxes_array[i][j], 1) / 20)

print(boxes_array) # all detected boxes, xmin ymin xmax ymax


WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)
CHARACTER_COLOR = (0, 0, 255)

class Spot:
	def __init__(self, row, col, width, total_rows):
		self.row = row
		self.col = col
		self.x = row * width
		self.y = col * width
		self.color = WHITE
		self.neighbors = []
		self.width = width
		self.total_rows = total_rows

	def get_pos(self):
		return self.row, self.col

	def is_closed(self):
		return self.color == RED

	def is_open(self):
		return self.color == GREEN

	def make_character(self):
		self.color = CHARACTER_COLOR

	def is_barrier(self):
		return self.color == BLACK

	def is_start(self):
		return self.color == ORANGE

	def is_end(self):
		return self.color == TURQUOISE

	def reset(self):
		self.color = WHITE

	def make_start(self):
		self.color = ORANGE

	def make_closed(self):
		self.color = RED

	def make_open(self):
		self.color = GREEN

	def make_barrier(self):
		self.color = BLACK

	def make_end(self):
		self.color = TURQUOISE

	def make_path(self):
		self.color = PURPLE

	def draw(self, win):
		pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

	def update_neighbors(self, grid, car_length, car_width):
		self.neighbors = []
		rows = len(grid)
		cols = len(grid[0]) if rows > 0 else 0

		directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # down, up, right, left
		for dx, dy in directions:
			new_row = self.row + dx
			new_col = self.col + dy

			if 0 <= new_row < rows and 0 <= new_col < cols and not grid[new_row][new_col].is_barrier():
				can_fit = True
				for i in range(-car_length // 2, car_length // 2 + 1):
					for j in range(-car_width // 2, car_width // 2 + 1):
						check_row = new_row + i
						check_col = new_col + j
						if not (0 <= check_row < rows and 0 <= check_col < cols and not grid[check_row][check_col].is_barrier()):
							can_fit = False
							break
					if not can_fit:
						break

				if can_fit:
					self.neighbors.append(grid[new_row][new_col])

	def __lt__(self, other):
		return False


def h(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)


def reconstruct_path(came_from, current, draw):
	while current in came_from:
		current = came_from[current]
		current.make_path()
		draw()


def algorithm(draw, grid, start, end):
	count = 0
	open_set = PriorityQueue()
	open_set.put((0, count, start))
	came_from = {}
	g_score = {spot: float("inf") for row in grid for spot in row}
	g_score[start] = 0
	f_score = {spot: float("inf") for row in grid for spot in row}
	f_score[start] = h(start.get_pos(), end.get_pos())

	open_set_hash = {start}

	while not open_set.empty():
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()

		current = open_set.get()[2]
		open_set_hash.remove(current)

		if current == end:
			reconstruct_path(came_from, end, draw)
			end.make_end()
			return True

		for neighbor in current.neighbors:
			temp_g_score = g_score[current] + 1

			if temp_g_score < g_score[neighbor]:
				came_from[neighbor] = current
				g_score[neighbor] = temp_g_score
				f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
				if neighbor not in open_set_hash:
					count += 1
					open_set.put((f_score[neighbor], count, neighbor))
					open_set_hash.add(neighbor)
					neighbor.make_open()

		draw()

		if current != start:
			current.make_closed()

	return False


def make_grid(rows, width):
	grid = []
	gap = width // rows
	for i in range(rows):
		grid.append([])
		for j in range(rows):
			spot = Spot(i, j, gap, rows)
			grid[i].append(spot)

	return grid


def draw_grid(win, rows, width):
	gap = width // rows
	for i in range(rows):
		pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
		for j in range(rows):
			pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win, grid, rows, width):
	win.fill(WHITE)

	for row in grid:
		for spot in row:
			spot.draw(win)

	draw_grid(win, rows, width)
	pygame.display.update()


def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y, x = pos

	row = y // gap
	col = x // gap

	return row, col

def create_barriers_from_list(grid, obstacles):
    for obstacle in obstacles:
        xmin, ymin, xmax, ymax = obstacle
        for row in range(int(xmin), int(xmax + 1)):
            for col in range(int(ymin), int(ymax + 1)):
                spot = grid[row][col]
                spot.make_barrier()


def main(win, width):
    
	CAR_LENGTH = 4
	CAR_WIDTH = 2
	ROWS = 50
	grid = make_grid(ROWS, width)

	start = None
	end = None
	character = None

	run = True

	create_barriers_from_list(grid, boxes_array)
	while run:
		draw(win, grid, ROWS, width)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

			if pygame.mouse.get_pressed()[0]: # LEFT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				spot = grid[row][col]
				if not start and spot != end:
					start = spot
					start.make_start()

				elif not end and spot != start:
					end = spot
					end.make_end()

				elif spot != end and spot != start:
					spot.make_barrier()

			elif pygame.mouse.get_pressed()[2]: # RIGHT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				spot = grid[row][col]
				spot.reset()
				if spot == start:
					start = None
				elif spot == end:
					end = None

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE and start and end:
					for row in grid:
						for spot in row:
							spot.update_neighbors(grid, CAR_LENGTH, CAR_WIDTH)

					algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
					character = start
					character.make_character()

				if event.key == pygame.K_c:
					start = None
					end = None
					grid = make_grid(ROWS, width)

				if character:
					row, col = character.get_pos()
					if event.key == pygame.K_a and col > 0 and not grid[row + 1][col].is_barrier():
						grid[row][col].reset() 
						character = grid[row - 1][col]
					elif event.key == pygame.K_d and col < ROWS - 1 and not grid[row + 1][col].is_barrier():
						grid[row][col].reset()
						character = grid[row + 1][col]  
					elif event.key == pygame.K_w and row > 0 and not grid[row][col - 1].is_barrier():
						grid[row][col].reset()
						character = grid[row][col - 1]
					elif event.key == pygame.K_s and row < ROWS - 1 and not grid[row][col + 1].is_barrier():
						grid[row][col].reset()
						character = grid[row][col + 1]
					character.make_character()
		if character and grid[character.row][character.col].is_barrier():
			print("COLLISION")

	pygame.quit()

main(WIN, WIDTH)