import heapq
import math

from utils import *
from collections import deque
from queue import PriorityQueue
from grid import Grid
from spot import Spot
from math import sqrt

def bfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Breadth-First Search (BFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    queue = deque([start])
    visited = {start}
    came_from = {}

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = queue.popleft()

        if current == end:
            # reconstruct the path
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def dfs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    Depdth-First Search (DFS) Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    stack = [start]
    visited = {start}
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = stack.pop()

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append(neighbor)
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def h_manhattan_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Manhattan distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    l = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])

    return l

def h_euclidian_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """
    Heuristic function for A* algorithm: uses the Euclidian distance between two points.
    Args:
        p1 (tuple[int, int]): The first point (x1, y1).
        p2 (tuple[int, int]): The second point (x2, y2).
    Returns:
        float: The Manhattan distance between p1 and p2.
    """
    l = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    return l


def astar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    """
    A* Pathfinding Algorithm.
    Args:
        draw (callable): A function to call to update the Pygame window.
        grid (Grid): The Grid object containing the spots.
        start (Spot): The starting spot.
        end (Spot): The ending spot.
    Returns:
        bool: True if a path is found, False otherwise.
    """
    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start))
    came_from = {}

    g_score = {spot: float('inf') for row in grid.grid for spot in row}
    g_score[start] = 0

    f_score = {spot: float('inf') for row in grid.grid for spot in row}
    f_score[start] = h_manhattan_distance(start.get_position(), end.get_position())

    open_set = {start}

    while not open_heap.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_heap.get()[2]
        open_set.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue

            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + h_manhattan_distance(neighbor.get_position(), end.get_position())
                if neighbor not in open_set:
                    count += 1
                    open_heap.put((f_score[neighbor], count, neighbor))
                    open_set.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def dls(draw: callable, grid: Grid, start: Spot, end: Spot, limit: int) -> bool:
    stack = [(start, 0)]
    visited = {start}
    came_from = {}

    while stack:
        current, depth = stack.pop()

        if depth > limit:
            continue

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                visited.add(neighbor)
                came_from[neighbor] = current
                stack.append((neighbor, depth + 1))
                neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def ucs(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:

    count = 0
    open_heap = PriorityQueue()
    open_heap.put((0, count, start))

    came_from = {}
    cost_so_far = {spot: float("inf") for row in grid.grid for spot in row}
    cost_so_far[start] = 0

    open_set = {start}

    while not open_heap.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current_cost, _, current = open_heap.get()
        open_set.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue

            new_cost = cost_so_far[current] + 1

            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current

                if neighbor not in open_set:
                    count += 1
                    open_heap.put((new_cost, count, neighbor))
                    open_set.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False


def dijkstra(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))

    came_from = {}
    distance = {spot: float("inf") for row in grid.grid for spot in row}
    distance[start] = 0

    pq_set = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current_distance, _, current = open_set.get()
        pq_set.remove(current)

        if current == end:
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            end.make_end()
            start.make_start()
            return True

        for neighbor in current.neighbors:
            if neighbor.is_barrier():
                continue

            weight = getattr(neighbor, "weight", 1)
            new_distance = distance[current] + weight

            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                came_from[neighbor] = current

                if neighbor not in pq_set:
                    count += 1
                    open_set.put((new_distance, count, neighbor))
                    pq_set.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False

def iddfs(draw: callable, grid: Grid, start: Spot, end: Spot,max_depth=None) -> bool:
    if max_depth is None:
        max_depth = grid.rows * grid.cols

    def dls(current, depth, visited, came_from):
        """Depth-Limited Search helper."""
        if depth < 0:
            return False
        visited.add(current)

        if current == end:
            return True

        for neighbor in current.neighbors:
            if neighbor not in visited and not neighbor.is_barrier():
                neighbor.make_open()
                draw()
                came_from[neighbor] = current
                if dls(neighbor, depth - 1, visited, came_from):
                    return True

        if current != start:
            current.make_closed()
            draw()
        return False

    for depth in range(max_depth):
        visited = set()
        came_from = {}
        if dls(start, depth, visited, came_from):
            current = end
            while current in came_from:
                current = came_from[current]
                current.make_path()
                draw()
            start.make_start()
            end.make_end()
            return True

    return False

def idastar(draw: callable, grid: Grid, start: Spot, end: Spot) -> bool:
    def heuristic(a, b):
        # Manhattan distance
        return abs(a.row - b.row) + abs(a.col - b.col)

    def search(path, g, threshold):
        current = path[-1]
        f = g + heuristic(current, end)

        if f > threshold:
            return f  # exceeded cost threshold

        if current == end:
            return True

        min_threshold = math.inf
        for neighbor in current.neighbors:
            if neighbor not in path and not neighbor.is_barrier():
                neighbor.make_open()
                draw()
                path.append(neighbor)
                temp = search(path, g + 1, threshold)
                if temp is True:
                    return True
                if temp < min_threshold:
                    min_threshold = temp
                path.pop()

        if current != start:
            current.make_closed()
            draw()
        return min_threshold

    threshold = heuristic(start, end)
    path = [start]

    while True:
        temp = search(path, 0, threshold)
        if temp is True:
            for spot in path:
                spot.make_path()
                draw()
            start.make_start()
            end.make_end()
            return True
        if temp == math.inf:
            return False
        threshold = temp

# and the others algorithms...
# ▢ Depth-Limited Search (DLS)
# ▢ Uninformed Cost Search (UCS)
# ▢ Greedy Search
# ▢ Iterative Deepening Search/Iterative Deepening Depth-First Search (IDS/IDDFS)
# ▢ Iterative Deepening A* (IDA)
# Assume that each edge (graph weight) equalss