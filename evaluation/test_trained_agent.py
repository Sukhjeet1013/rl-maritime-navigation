from stable_baselines3 import PPO
from env.ship_navigation_env import ShipNavigationEnv


def evaluate():

    env = ShipNavigationEnv()

    model = PPO.load("models/ship_navigation_model")

    episodes = 20

    goals = 0
    crashes = 0

    print("\nStarting Evaluation...\n")

    for episode in range(1, episodes + 1):

        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:

            action, _ = model.predict(state, deterministic=True)

            state, reward, done, truncated, _ = env.step(action)

            total_reward += reward
            steps += 1

        if reward >= 100:
            goals += 1
            print(f"Episode {episode}: GOAL reached in {steps} steps")

        else:
            crashes += 1
            print(f"Episode {episode}: SHIP crashed after {steps} steps")

    print("\nEvaluation Summary")
    print("-------------------")
    print(f"Episodes: {episodes}")
    print(f"Goals: {goals}")
    print(f"Crashes: {crashes}")
    print(f"Success Rate: {(goals / episodes) * 100:.2f}%\n")


if __name__ == "__main__":
    evaluate()