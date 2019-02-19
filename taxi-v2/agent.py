import numpy as np
from collections import defaultdict
import random


class Agent:

    def __init__(self, num_episodes, nA=6, alpha=0.1, gamma=1.0, eps=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.i_episode = 1
        self.num_episodes = num_episodes
        self.eps = eps
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.epsilon_greedy(state)

    def epsilon_greedy(self, state):
        if random.random() > self.eps:
            q_s = self.Q[state]
            action = np.argmax(q_s)
        else:
            action = random.randint(0, self.nA - 1)
        return action

    def update_hyper_parameter(self):
        self.i_episode += 1
        if self.i_episode <= 0.5 * self.num_episodes:
            self.eps -= (1.0/self.i_episode)
            self.gamma = 0.9
            self.alpha = 0.05
        elif self.i_episode <= 0.8*self.num_episodes:
            self.eps = 0.5/(self.i_episode-self.num_episodes*0.5)
            self.gamma = 0.8
            self.alpha = 0.01
        else:
            self.eps = 0.0
            self.gamma = 0.9
            self.alpha = 0.2
        return

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = self.update_q_learning(state, action, reward, next_state)
        if done:
            self.update_hyper_parameter()

        return

    def update_q_learning(self, state, action, reward, next_state=None):
        q_s_a = self.Q[state][action]
        next_q_s_max = np.max(self.Q[next_state]) if next_state is not None else 0
        return q_s_a + self.alpha * (reward + self.gamma * next_q_s_max - q_s_a)

