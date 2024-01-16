from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
import gym

# Create the environment
env = gym.make("Taxi-v3", render_mode="rgb_array")
env.reset()

# Initialize the model
model = A2C("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=250000)

# Save the trained model
model.save("a2c_taxi_2")

# Close the environment
env.close()