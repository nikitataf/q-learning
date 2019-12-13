import gym
import random
import os
import numpy as np
import time
from time import sleep


def clear():
    os.system('clear')


def print_frames(frames):
    for i, frame in enumerate(frames):
        clear()
        print(frame['frame'].getvalue())
        print(f"Time step: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        sleep(.1)


def brute_force():
    """Brute force solution
    :return:
    """
    env.reset()
    env.s = 1  # set environment to illustration's state
    penalties, reward, epochs, total_reward = 0, 0, 0, 0
    frames = []  # for animation
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if reward == -10:
            penalties += 1
        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward})
        epochs += 1
        total_reward += reward

    print_frames(frames)
    print("Actions taken: {}".format(epochs))
    print("Total reward: {}".format(total_reward))
    print("Penalties incurred: {}".format(penalties))


def training(args):
    """Q-learning
    :param args: (alpha, gamma, epsilon).
    :return:
    """
    (alpha, gamma, epsilon) = args
    q_table = np.zeros([env.observation_space.n, env.action_space.n])  # 500×6  matrix of zeros

    for i in range(0, 50000):
        state = env.reset()
        epochs, penalties, reward, = 0, 0, 0
        done = False
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values
            next_state, reward, done, info = env.step(action)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            if reward == -10:
                penalties += 1
            state = next_state
            epochs += 1

        if i % 1000 == 0:
            print(f"Training: {i}")

    print("Training finished.\n")
    return q_table


def evaluation(q_table, show_animation=False):
    """Evaluate agent's performance after Q-learning"""

    total_epochs, total_penalties, total_reward = 0, 0, 0
    episodes = 10
    frames = []

    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        done = False

        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)
            if reward == -10:
                penalties += 1
            # Put each rendered frame into dict for animation
            frames.append({
                'frame': env.render(mode='ansi'),
                'state': state,
                'action': action,
                'reward': reward})
            total_reward += reward
            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    if show_animation:
        print_frames(frames)
    print(f"Results after {episodes} episodes:")
    print(f"Average total reward: {total_reward / episodes}")
    print(f"Average number of actions: {total_epochs / episodes}")
    print(f"Average penalties: {total_penalties / episodes}")


if __name__ == "__main__":
    """
    Actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters: locations
    """

    # Make environment
    env = gym.make("Taxi-v2")

    # Brute force approach
    brute_force()

    # Q-learning
    alpha = 0.1  # learning rate / step size
    gamma = 0.9  # discount factor
    epsilon = 0.1
    args = (alpha, gamma, epsilon)

    q_table = training(args)
    evaluation(q_table, show_animation=True)
