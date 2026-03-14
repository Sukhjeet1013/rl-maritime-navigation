from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.ship_navigation_env import ShipNavigationEnv


def train():

    env = DummyVecEnv([lambda: ShipNavigationEnv()])

    model = PPO(
        "MlpPolicy",
        env,

        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,

        verbose=1,
        device="cpu"
    )

    model.learn(total_timesteps=4000000)

    model.save("models/ship_navigation_model")


if __name__ == "__main__":
    train()