import gymnasium as gym
from stable_baselines3 import PPO

# render_mode="human" active la fenêtre 3D
env = gym.make("Ant-v4", render_mode="human")

# Initialisation de l'agent PPO
model = PPO("MlpPolicy", env, verbose=1)

# Lancement de l'apprentissage
print("Début de l'entraînement...")
model.learn(total_timesteps=50000)

env.close()
