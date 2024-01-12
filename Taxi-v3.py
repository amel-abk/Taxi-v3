import gym
import numpy as np

# Create the Taxi-v3 environment
env = gym.make("Taxi-v3", render_mode="human")

# Initialize Q-table with zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.01  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration-exploitation 

# Number of episodes for training
num_episodes = 10000

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    
    while not done:
        # Exploration-exploitation
        if np.random.uniform(0, 1) < epsilon:
            # Explore: choose a random action
            action = env.action_space.sample()  
        else:
            # Exploit: choose the action with the highest Q-value
            action = np.argmax(Q[state])  
        
        next_state, reward, done, _, _ = env.step(action)  # Take the chosen action
        
        # Update Q-value using the Q-learning update rule
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
        
        total_reward += reward
        state = next_state
        #env.render() 
    
    # Print total reward for the episode
    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")


# Evaluate the trained agent

num_eval_episodes = 10

for _ in range(num_eval_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0 
    
    while not done:
        action = np.argmax(Q[state])  
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        state = next_state
        #env.render() 
    
    print(f'Episode Reward: {episode_reward}')

# Close the environment
env.close()
