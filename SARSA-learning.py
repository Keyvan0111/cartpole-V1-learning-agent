"""
The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import gym
import numpy as np
import random

class SARSA_Agent():
    def __init__(self):
        self.env = gym.make('CartPole-v1', render_mode=None) # set rendermode to 'human' to get visuals

        self.num_episodes = 10000 
        self.max_steps = 100
        
        # Rules for the cartpole
        self.pos_range = np.linspace(-2.4, 2.4, 10)
        self.vel_range = np.linspace(-5, 5, 10)
        self.angle_range = np.linspace(-.2095, .2095, 10)
        self.angle_vel_range = np.linspace(-5, 5, 10)

        self.rewards_all_episodes = []
        self.q_table = np.zeros((len(self.pos_range)+1, len(self.vel_range)+1, len(self.angle_range)+1, len(self.angle_vel_range)+1, self.env.action_space.n))

        self.learning_rate = 0.1 # also known as alpha
        self.discount_rate = 0.99 # also known as Gamma
        self.max_epsilon = 1
        self.epsilon = self.max_epsilon
        self.min_epsilon = 0.05
        self.decay_epsilon = 0.001

    def choose_action(self, states):
        if np.random.uniform(0,1) > self.epsilon:
            action = np.argmax(self.q_table[states[0], states[1], states[2], states[3], :])
        else:
            action = self.env.action_space.sample()
        return action
    
    def update_Qvalue(self, action, new_action, reward, states, new_states):

        self.q_table[states[0], states[1], states[2], states[3], action] = self.q_table[states[0], states[1], states[2], states[3], action] + \
        self.learning_rate * (reward + self.discount_rate * self.q_table[new_states[0], new_states[1], new_states[2], new_states[3], new_action] - \
        self.q_table[states[0], states[1], states[2], states[3], action])
        return

    def status_print(self, rewards, episodeNumber):
        """
        Prints the status of the training process including the episode number,
        current epsilon value, and the mean reward of the last 100 episodes.

        Parameters:
        - rewards: The total rewards obtained in the current episode.
        - episodeNumber: The current episode number.

        Returns:
        - None
        """
        self.rewards_all_episodes.append(rewards)
        mean_rewards = np.mean(self.rewards_all_episodes[-100:])
        if episodeNumber % 50 == 0:
            print(f'Episode: {episodeNumber}  Epsilon: {self.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
    
    def train_model(self):


        for episode in range(self.num_episodes):
            state = self.env.reset()[0]
            spaces = [self.pos_range, self.vel_range, self.angle_range, self.angle_vel_range]
            state_discretized = [np.digitize(state[i], spaces[i]) for i in range(len(state))]
            initail_action = self.choose_action(state_discretized)
            done = False
            rewards = 0

            for _ in range(self.max_steps):

                new_state, reward, done, _, _ = self.env.step(initail_action)
                new_state_discretized = [np.digitize(new_state[i], spaces[i]) for i in range(len(new_state))]
                new_action = self.choose_action(new_state_discretized)
                self.update_Qvalue(initail_action, new_action, reward, state_discretized, new_state_discretized)               

                state = new_state
                state_discretized = [new_state_discretized[i] for i in range(len(new_state_discretized))]
                initail_action = new_action
                rewards += reward

                self.status_print(rewards, episode)

                if done == True:
                    break

            self.status_print(rewards, episode)

            self.epsilon = max(self.epsilon - self.decay_epsilon, 0)
            
        self.env.close()

if __name__ == "__main__":
    agent = SARSA_Agent()
    agent.train_model()
    