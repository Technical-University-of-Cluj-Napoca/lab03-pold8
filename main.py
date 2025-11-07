from utils import *
from grid import Grid
from searching_algorithms import *
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

BUTTON_HEIGHT = 40
BUTTON_WIDTH = (WIDTH - 10 * 9) // 8
BUTTON_Y = HEIGHT + 10
TOTAL_HEIGHT = HEIGHT + BUTTON_HEIGHT + 20

BUTTONS_CONFIG = [
    {"text": "BFS", "key": pygame.K_1, "func": bfs},
    {"text": "DFS", "key": pygame.K_2, "func": dfs},
    {"text": "A*", "key": pygame.K_3, "func": astar},
    {"text": "DLS", "key": pygame.K_4,
     "func": lambda draw_func, grid, start, end: dls(draw_func, grid, start, end, 1000)},
    {"text": "UCS", "key": pygame.K_5, "func": ucs},
    {"text": "Dijkstra", "key": pygame.K_6, "func": dijkstra},
    {"text": "IDDFS", "key": pygame.K_7, "func": iddfs},
    {"text": "IDA*", "key": pygame.K_8, "func": idastar},
]

BUTTON_RECS = []
for i, config in enumerate(BUTTONS_CONFIG):
    x = 10 + i * (BUTTON_WIDTH + 10)
    rect = pygame.Rect(x, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT)
    BUTTON_RECS.append(rect)
    config["rect"] = rect


def draw_button(win, button_config):
    rect = button_config["rect"]
    text = button_config["text"]

    pygame.draw.rect(win, GREEN, rect, border_radius=5)

    text_surface = FONT.render(text, True, BLACK)
    text_rect = text_surface.get_rect(center=rect.center)
    win.blit(text_surface, text_rect)


if __name__ == "__main__":
    pygame.init()

    WIN = pygame.display.set_mode((WIDTH, TOTAL_HEIGHT))
    pygame.display.set_caption("Path Visualizing Algorithm")

    try:
        FONT = pygame.font.SysFont('arial', 14, bold=True)
    except:
        FONT = pygame.font.Font(None, 20)

    clock = pygame.time.Clock()
    FPS = 60

    ROWS = 50
    COLS = 50
    grid = Grid(WIN, ROWS, COLS, WIDTH, HEIGHT)

    start = None
    end = None
    run = True
    started = False

    while run:
        WIN.fill(WHITE)
        grid.draw()

        for config in BUTTONS_CONFIG:
            draw_button(WIN, config)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()

                if pos[1] < HEIGHT:
                    row, col = grid.get_clicked_pos(pos)

                    if row >= ROWS or row < 0 or col >= COLS or col < 0:
                        continue

                    spot = grid.grid[row][col]
                    if not start and spot != end:
                        start = spot
                        start.make_start()
                    elif not end and spot != start:
                        end = spot
                        end.make_end()
                    elif spot != end and spot != start:
                        spot.make_barrier()

                else:
                    for config in BUTTONS_CONFIG:
                        if config["rect"].collidepoint(pos):
                            print(f"Running {config['text'].split()[0]}...")
                            started = True
                            for row in grid.grid:
                                for spot in row:
                                    spot.update_neighbors(grid.grid)

                            algorithm_func = config["func"]


                            draw_func = lambda: (grid.draw(), pygame.display.update(), pygame.time.delay(1))

                            found = algorithm_func(draw_func, grid, start, end)

                            algo_name = config['text'].split()[0]
                            if found:
                                print(f"Path found with {algo_name}!")
                            else:
                                print(f"No path found with {algo_name}.")
                            started = False
                            break

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                if pos[1] < HEIGHT:
                    row, col = grid.get_clicked_pos(pos)
                    spot = grid.grid[row][col]
                    spot.reset()

                    if spot == start:
                        start = None
                    elif spot == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    print("Clearing the grid...")
                    start = None
                    end = None
                    grid.reset()

        # Ensures the main loop stays at a steady FPS when no algorithm is running
        clock.tick(FPS)

    pygame.quit()