import pygame
from stable_baselines3 import PPO
from env.ship_navigation_env import ShipNavigationEnv


CELL_SIZE = 60
GRID_SIZE = 10
WINDOW_SIZE = GRID_SIZE * CELL_SIZE


WHITE = (255,255,255)
OCEAN = (12,45,75)
SHIP = (40,200,255)
ROCK = (120,120,120)
GOAL = (60,220,100)
CYAN = (0,255,255)


def draw_grid(screen):

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):

            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )

            pygame.draw.rect(screen, WHITE, rect, 1)


def draw_ship(screen, pos):

    x = int(pos[0] * CELL_SIZE)
    y = int(pos[1] * CELL_SIZE)

    center_x = x + CELL_SIZE // 2
    center_y = y + CELL_SIZE // 2

    size = int(CELL_SIZE * 0.35)

    points = [
        (center_x, center_y - size),
        (center_x - size, center_y + size),
        (center_x + size, center_y + size)
    ]

    pygame.draw.polygon(screen, SHIP, points)


def draw_goal(screen, pos):

    x = int(pos[0] * CELL_SIZE)
    y = int(pos[1] * CELL_SIZE)

    pygame.draw.rect(screen, GOAL, (x, y, CELL_SIZE, CELL_SIZE))


def draw_obstacles(screen, obstacles):

    for obs in obstacles:

        pos = obs["pos"]

        x = int(pos[0] * CELL_SIZE)
        y = int(pos[1] * CELL_SIZE)

        center_x = x + CELL_SIZE // 2
        center_y = y + CELL_SIZE // 2

        pygame.draw.circle(
            screen,
            ROCK,
            (center_x, center_y),
            int(CELL_SIZE * 0.4)
        )


def draw_current(screen, current):

    start_x = WINDOW_SIZE - 90
    start_y = 80

    start = (start_x, start_y)

    end = (
        start_x + int(current[0]) * 40,
        start_y + int(current[1]) * 40
    )

    pygame.draw.line(screen, CYAN, start, end, 4)
    pygame.draw.circle(screen, CYAN, start, 6)


def run_simulation():

    pygame.init()

    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Autonomous Maritime Navigation")

    env = ShipNavigationEnv()
    model = PPO.load("models/ship_navigation_model")

    font = pygame.font.SysFont(None, 24)

    state, _ = env.reset()

    clock = pygame.time.Clock()
    running = True

    episode = 1
    goals = 0
    crashes = 0

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        old_ship = env.ship_pos.copy()
        old_obstacles = [obs["pos"].copy() for obs in env.obstacles]

        action, _ = model.predict(state, deterministic=True)

        state, reward, done, truncated, _ = env.step(action)

        new_ship = env.ship_pos.copy()
        new_obstacles = [obs["pos"].copy() for obs in env.obstacles]

        animation_steps = 16

        for i in range(animation_steps):

            t = i / animation_steps

            ship_x = old_ship[0] + (new_ship[0] - old_ship[0]) * t
            ship_y = old_ship[1] + (new_ship[1] - old_ship[1]) * t

            screen.fill(OCEAN)

            draw_grid(screen)
            draw_ship(screen, (ship_x, ship_y))
            draw_goal(screen, env.goal_pos)

            # smooth obstacle interpolation
            interpolated_obstacles = []

            for j in range(len(new_obstacles)):

                ox = old_obstacles[j][0] + (new_obstacles[j][0] - old_obstacles[j][0]) * t
                oy = old_obstacles[j][1] + (new_obstacles[j][1] - old_obstacles[j][1]) * t

                interpolated_obstacles.append({"pos": (ox, oy)})

            draw_obstacles(screen, interpolated_obstacles)

            draw_current(screen, env.current)

            # draw stats
            stats = font.render(
                f"Episode {episode}  Goals:{goals}  Crashes:{crashes}",
                True,
                WHITE
            )

            screen.blit(stats, (10,10))

            pygame.display.update()

            pygame.time.delay(35)

        clock.tick(2)

        if done:

            if reward >= 100:
                goals += 1
                print(f"Episode {episode}: GOAL reached")

            else:
                crashes += 1
                print(f"Episode {episode}: SHIP crashed")

            print(f"Goals: {goals} | Crashes: {crashes}\n")

            episode += 1

            state, _ = env.reset()

    pygame.quit()


if __name__ == "__main__":
    run_simulation()