from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt


def plot_performance(num_episodes, avg_score, plot_every):
    plt.plot(np.linspace(0, num_episodes, len(avg_score), endpoint=False), np.asarray(avg_score))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next {} Episodes)'.format(plot_every))
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_score))
    return


def main():
    env = gym.make('Taxi-v2')
    num_episodes = 20000
    window = 100
    agent = Agent(num_episodes)
    avg_rewards, best_avg_reward = interact(env, agent, num_episodes=num_episodes, window=window)

    # print('avg_rewards: {}, best_avg_reward: {}'.format(avg_rewards, best_avg_reward))
    plot_performance(num_episodes=num_episodes, avg_score=avg_rewards, plot_every=window)
    return


if __name__ == '__main__':
    main()

