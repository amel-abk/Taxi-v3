from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
import gym

# Create the environment
env = gym.make("Taxi-v3", render_mode="rgb_array")
env.reset()

# load the model
model = A2C.load("a2c_taxi_2")

# Evaluate the trained model
num_eval_episodes = 10
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_eval_episodes, deterministic=True)

print(f"mean_reward={mean_reward}")
# Close the environment
env.close() 