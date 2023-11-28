import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', is_slippery=False)

total_actions = env.action_space.n
assert(total_actions == 4), f"There are a total of four possible actions in this environment. Your answer is {total_actions}"

def random_action(env):
    return np.random.randint(env.action_space.n)

observation, info = env.reset()

# Performing an action
action = random_action(env)
observation, reward, done, _, info = env.step(action)

# Displaying the first frame of the game
plt.imshow(env.render())

# Printing game info
print(f"actions: {env.action_space.n}\nstates: {env.observation_space.n}")
print(f"Current state: {observation}")

# Closing the environment
env.close()