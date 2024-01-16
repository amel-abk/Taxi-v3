import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

# Créer l'environnement CartPole
env = gym.make('CartPole-v1')

# Créer et entrainer le modèle PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Sauvegarder le modèle
model.save("ppo_cartpole")

# Charger le modèle
loaded_model = PPO.load("ppo_cartpole")

for i in range(10):
    obs, _ = env.reset()
    total_reward = 0

    while True:
        action, _ = loaded_model.predict(obs)
        print(f'Episode {i} Action: {action}')
        obs, reward, done, _, _ = env.step(int(action))
        total_reward += reward

        if done:
            print(f'Episode {i} Reward: {total_reward}')
            break


# Fermer l'environnement
env.close()