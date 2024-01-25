import gym
from gym import spaces
import numpy as np
import copy 


class WaterColorPuzzleEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, num_vials, num_colors, water_capacity):
        super(WaterColorPuzzleEnv, self).__init__()
        self.num_vials = num_vials
        self.num_colors = num_colors
        self.water_capacity = water_capacity

        # Define the action space
        self.action_space = spaces.Discrete(num_vials * num_colors)
        # Define the observation space
        self.observation_space = spaces.Discrete(self.num_vials * self.num_colors)
        # Initialize other variables
        self.max_steps = 500
        self.current_step = 0
        

    def _generate_initial_state(self):
        observation = []
        for i in range(1, self.num_vials):
            for j in range(self.num_colors):
                observation.append(i)

        np.random.shuffle(observation)
        observation = observation + [0] * (self.num_vials * self.water_capacity - len(observation))

        return np.array(observation)


    def step(self, action):
        # Convert the action into pairs of source and destination vials
        source_vial, destination_vial = self._action_to_vials(action)
      
        # Check if it is possible to pour from the source vial to the destination vial
        if self._can_pour_from_vial(source_vial) and self._can_pour_to_vial(destination_vial):
            # Perform the pouring action
            observation , reward= self._pour_color_action(source_vial, destination_vial)
            print('observation_from_step', observation)

            # Calculate the reward 
            self.total_reward += reward

            # Check if the game is over
            done = self._check_game_over()
            
            # Update the state
            self.state = observation


        else:
            print("Invalid source or destination vial index:", source_vial, destination_vial)
            self.total_reward -= 1
            return self.state, self.total_reward, False, {}
        
        # Update the step counter
        self.current_step += 1

        # Check if the maximum number of steps has been reached
        if self.current_step >= self.max_steps:
            done = True

        # Return the necessary information to the agent
        return self.state, self.total_reward, done, {}


    def reset(self):
        self.state = self._generate_initial_state()
        self.current_step = 0
        self.total_reward= 0
        return self.state


    def _pour_color_action(self, source_vial, destination_vial):
        s_vial = [self.state[source_vial + i] for i in range(self.water_capacity)]
        d_vial = [self.state[destination_vial + i] for i in range(self.water_capacity)]
        last_non_zero_index = len(s_vial) - 1 - s_vial[::-1].index(next((color for color in s_vial[::-1] if color != 0)))
        source_color = s_vial[last_non_zero_index]
        

        # Find the color before zero in the destination
        color_before_zero = None
        for color in reversed(d_vial):
            if color != 0:
                color_before_zero = color
                break
                
        destination_index = d_vial.index(0)
        #Perform the permutation
        new_state = np.copy(self.state)
        
        new_state[source_vial + last_non_zero_index], new_state[destination_vial + destination_index] = new_state[destination_vial + destination_index], source_color
        print('new_state_from_pour_fonction:', new_state)
        # Compare the source color after the permutation with the color before zero
        if source_color == color_before_zero or color_before_zero == None:
            return new_state, 1
        else:
            # No color match, assign a reward of -1
            return self.state, -1
    
    def _get_destination_vials(self):
        destination_vials = [i for i, colors in enumerate(self.state) if 0 in colors]
        return destination_vials[0]


    # Check if all vials have only one color
    def _check_game_over(self):
        return np.all(self.state == 1)  


    # Check if there is an empty slot in the vial
    def _can_pour_to_vial(self, vial):
        return any(color == 0 for color in self.state[vial:vial + self.num_colors])


    # Check if the source is not empty
    def _can_pour_from_vial(self, vial):
        vial_contents = self.state[vial:vial+ self.num_colors]
        return any(color != 0 for color in vial_contents)


    def _action_to_vials(self, action):
        source_vial = action % self.num_vials
        destination_vial = ((action // self.num_vials) + 1 + source_vial) % self.num_vials
        return source_vial * 3, destination_vial * 3

# Create an instance of the environment
env = WaterColorPuzzleEnv(num_vials=4, num_colors=3, water_capacity=3)

# Reset the environment
observation = env.reset()
print("initial state:", observation)

for _ in range(10):
    # Choose a random action
    action = env.action_space.sample()

    # Display the indices of source and destination vials associated with each action
    source_vial, destination_vial = env._action_to_vials(action)
    print("Action:", action, "Source Vial:", source_vial, "Destination Vial:", destination_vial)

    observation, reward, done, info = env.step(action)

    print("next state:", observation)
    print("Reward:", reward)
    print("done:", done)

    if done:
        break