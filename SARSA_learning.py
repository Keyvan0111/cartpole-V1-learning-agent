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
    """
    A SARSA agent for solving the CartPole-v1 problem using the State-Action-Reward-State-Action (SARSA) learning algorithm.

    Attributes:
        - env (gym.Env):  The Gym environment.
        - num_episodes (int):  The number of episodes to train the agent.
        - max_steps (int):  The maximum number of steps per episode.
        - pos_range (numpy.ndarray):  The discretized range for the cart position.
        - vel_range (numpy.ndarray):  The discretized range for the cart velocity.
        - angle_range (numpy.ndarray):  The discretized range for the pole angle.
        - angle_vel_range (numpy.ndarray):  The discretized range for the pole angular velocity.
        - rewards_all_episodes (list):  A list to store rewards from all episodes.
        - q_table (numpy.ndarray):  The Q-table for storing state-action values.
        - learning_rate (float):  The learning rate (alpha).
        - discount_rate (float):  The discount rate (gamma).
        - max_epsilon (float):  The maximum epsilon for epsilon-greedy action selection.
        - epsilon (float):  The current epsilon value for epsilon-greedy action selection.
        - min_epsilon (float):  The minimum epsilon value for epsilon decay.
        - decay_epsilon (float):  The decay rate for epsilon.
    """
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
        """
        Chooses an action based on the current state using an epsilon-greedy policy.

        Parameters:
            states (list): The discretized representation of the current state.

        Returns:
            int: The action selected.
        """
        if np.random.uniform(0,1) > self.epsilon:
            action = np.argmax(self.q_table[states[0], states[1], states[2], states[3], :])
        else:
            action = self.env.action_space.sample()
        return action
    
    def update_Qvalue(self, action, new_action, reward, states, new_states):
        """
        Updates the Q-value for a given state-action pair using the SARSA update formula.

        Parameters:
            action (int): The current action taken.
            new_action (int): The next action selected according to the policy.
            reward (float): The reward received after taking the action.
            states (list): The discretized representation of the current state.
            new_states (list): The discretized representation of the next state.

        Returns:
            None
        """
        self.q_table[states[0], states[1], states[2], states[3], action] = self.q_table[states[0], states[1], states[2], states[3], action] + \
        self.learning_rate * (reward + self.discount_rate * self.q_table[new_states[0], new_states[1], new_states[2], new_states[3], new_action] - \
        self.q_table[states[0], states[1], states[2], states[3], action])
        return

    def status_print(self, rewards, episodeNumber):
        """
        Prints the training status, including the current episode number, epsilon value,
        and the mean reward of the last 100 episodes.

        Parameters:
            rewards (float): The total rewards obtained in the current episode.
            episodeNumber (int): The current episode number.

        Returns:
            None
        """
        self.rewards_all_episodes.append(rewards)
        mean_rewards = np.mean(self.rewards_all_episodes[-100:])
        if episodeNumber % 50 == 0:
            print(f'Episode: {episodeNumber}  Epsilon: {self.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')
    
    def train_model(self):
        """
        Trains the SARSA agent over a specified number of episodes.

        During training, the agent selects actions based on the current policy,
        updates the Q-values using the SARSA formula, and gradually decays epsilon.

        Returns:
            None
        """
        for episode in range(self.num_episodes):
            state = self.env.reset()[0]
            spaces = [self.pos_range, self.vel_range, self.angle_range, self.angle_vel_range]
            state_discretized = [np.digitize(state[i], spaces[i]) for i in range(len(state))]
            initail_action = self.choose_action(state_discretized) # set the policy
            done = False
            rewards = 0

            # training loop

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
