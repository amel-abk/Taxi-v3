import gym
import numpy as np
env = gym.make("Taxi-v3", render_mode='human').env
env.reset()
env.render()

'''print("Action Space{}".format(env.action_space))
print("State Space{}".format(env.observation_space))'''

#print("Action Space", (env.action_space.n))

'''env.s =114
env.render()


state, reward, done, info, _= env.step(1)
print('state:', state, 'reward:', reward,'done:', done,'info:', info)
env.render()'''


'''#randomly
state= env.reset()
counter=0
reward= None
while reward != 20:
    next_state, reward, done, info, _= env.step(env.action_space.sample())
    counter += 1
    env.render()

print(counter)'''


Q = np.zeros([env.observation_space.n, env.action_space.n])
G=0
alpha= 0.618


for episode in range(1, 1001):
    done= False
    G, reward= 0,0
    state, _= env.reset()
    #print(state)
    #break
    while done != True:
        action= np.argmax(Q[state])
        next_state, reward, done, info, _ = env.step(action)
        Q[state, action] += alpha * (reward + np.max([next_state])- Q[state, action])
        G += reward
        state = next_state
        env.render()
    if episode % 50 == 0:
        print('Episode {} Total Reward: {}'.format(episode, G))