from env.ship_navigation_env import ShipNavigationEnv

env = ShipNavigationEnv()

state, _ = env.reset()

print("Initial State:", state)

for _ in range(10):

    action = env.action_space.sample()

    state, reward, done, truncated, _ = env.step(action)

    print("State:", state, "Reward:", reward)

    if done:
        print("Goal reached!")
        break