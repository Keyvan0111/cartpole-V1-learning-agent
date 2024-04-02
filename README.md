# CartPole Q-Learning Agent

This repository contains a Python implementation of a Q-learning agent designed to solve the CartPole-v1 problem from OpenAI's Gym. The project aims to demonstrate the basic principles of Q-learning, a model-free reinforcement learning algorithm, by applying it to a classic control task.

## Project Overview

The CartPole problem is a well-known benchmark in the field of reinforcement learning. The challenge is to balance a pole on a cart by moving the cart left or right. The Q-learning agent learns to solve this task by discretizing the continuous state space of the CartPole environment and updating a Q-table based on the rewards received for its actions.

## Features

- Implementation of the Q-learning algorithm.
- Discretization of the continuous state space of the CartPole-v1 environment.
- Epsilon-greedy policy for action selection, balancing exploration and exploitation.
- Tracking and printing the mean rewards over episodes to monitor the agent's performance.

## Dependencies

- numpy
- gym

## Usage

To run the Q-learning agent on the CartPole-v1 problem, simply execute the `Q-learning.py` script:

```bash
python Q-learning.py
