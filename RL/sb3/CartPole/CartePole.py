import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

# Créer l'environnement CartPole
env = gym.make('CartPole-v1')

# Créer et entrainer le modèle PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)

# Sauvegarder le modèle
model.save("ppo_cartpole")

# Charger le modèle
loaded_model = PPO.load("ppo_cartpole")

# Evaluate the trained model
num_eval_episodes = 10
mean_reward, std_reward = evaluate_policy(loaded_model, env, n_eval_episodes=num_eval_episodes, deterministic=True)

print(f"mean_reward={mean_reward}")

# Fermer l'environnement
env.close()