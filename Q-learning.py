import numpy as np
import gym
import random


class Q_agent():
    def __init__(self):
        self.num_episodes = 10000
        self.max_steps = 100

        self.env = gym.make('CartPole-v1', render_mode=None)

        # Got some values from task 2 environment
        # linspace descretesizes the values
        self.pos_range = np.linspace(-2.4, 2.4, 10)
        self.vel_range = np.linspace(-5, 5, 10)
        self.angle_range = np.linspace(-.2095, .2095, 10)
        self.angle_vel_range = np.linspace(-5, 5, 10)

        self.q_table = np.zeros((len(self.pos_range)+1, len(self.vel_range)+1, len(self.angle_range)+1, len(self.angle_vel_range)+1, self.env.action_space.n))

        self.learning_rate = 0.1
        self.discount_rate = 0.99

        self.epsilon = 1
        self.max_epsilon = 1
        self.min_epsilon = 0.05
        self.decay_epsilon = 0.001
    
    def choose_action(self, state_discretized):
            exploration_rate_threshold = random.uniform(0,1)
            if exploration_rate_threshold > self.epsilon:
                action = np.argmax(self.q_table[state_discretized[0], state_discretized[1], state_discretized[2], state_discretized[3]])
            else:
                action = self.env.action_space.sample()
            return action
    
    def update_Qvalue(self, action, reward, state_discretized, new_state_discretized):
        self.q_table[state_discretized[0], state_discretized[1], state_discretized[2], state_discretized[3], action] = \
                self.q_table[state_discretized[0], state_discretized[1], state_discretized[2], state_discretized[3], action] + \
                self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state_discretized[0], new_state_discretized[1],\
                new_state_discretized[2], new_state_discretized[3],:]) - self.q_table[state_discretized[0],state_discretized[1],\
                state_discretized[2], state_discretized[3], action])
        return
    
    def status_print(self, rewards_all_episodes, rewards, episodeNumber):
        rewards_all_episodes.append(rewards)
        mean_rewards = np.mean(rewards_all_episodes[-100:])
        if episodeNumber % 50 == 0:
            print(f'Episode: {episodeNumber}  Epsilon: {self.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

    def train_model2(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()[0]
            rewards_all_episodes = []
            spaces = [self.pos_range, self.vel_range, self.angle_range, self.angle_vel_range]
            state_discretized = [np.digitize(state[i], spaces[i]) for i in range(len(state))]

            done = False
            rewards = 0
            for _ in range(self.max_steps):
                action = self.choose_action(state_discretized)

                new_state, reward, done , _, _ = self.env.step(action)
                new_state_discretized = [np.digitize(new_state[i], spaces[i]) for i in range(len(new_state))]

                self.update_Qvalue(action, reward, state_discretized, new_state_discretized)

                state = new_state
                state_discretized = [new_state_discretized[i] for i in range(len(new_state_discretized))]
                rewards+=reward

                if done == True:
                    break

            self.status_print(rewards_all_episodes, rewards, episode)
            # self.stats(episode)

            self.epsilon = max(self.epsilon - self.decay_epsilon, 0)
    def train_model(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()[0]
            rewards_all_episodes = []
            spaces = [self.pos_range, self.vel_range, self.angle_range, self.angle_vel_range]
            state_discretized = [np.digitize(state[i], spaces[i]) for i in range(len(state))]

            done = False
            rewards = 0
            for _ in range(self.max_steps):
                exploration_rate_threshold = random.uniform(0,1)
                if exploration_rate_threshold > self.epsilon:
                    action = np.argmax(self.q_table[state_discretized[0], state_discretized[1], state_discretized[2], state_discretized[3]])
                else:
                    action = self.env.action_space.sample()

                new_state, reward, done , _, _ = self.env.step(action)
                new_state_discretized = [np.digitize(new_state[i], spaces[i]) for i in range(len(new_state))]

                self.q_table[state_discretized[0], state_discretized[1], state_discretized[2], state_discretized[3], action] = \
                self.q_table[state_discretized[0], state_discretized[1], state_discretized[2], state_discretized[3], action] + \
                self.learning_rate * (reward + self.discount_rate * np.max(self.q_table[new_state_discretized[0], new_state_discretized[1],\
                new_state_discretized[2], new_state_discretized[3],:]) - self.q_table[state_discretized[0],state_discretized[1],\
                state_discretized[2], state_discretized[3], action]
                )

                state = new_state
                state_discretized = [new_state_discretized[i] for i in range(len(new_state_discretized))]
                rewards+=reward

                if done == True:
                    break
            rewards_all_episodes.append(rewards)
            mean_rewards = np.mean(rewards_all_episodes[len(rewards_all_episodes)-100:])
            if episode % 100 == 0:
                print(f'Episode: {episode} {rewards}  Epsilon: {self.epsilon:0.2f}  Mean Rewards {mean_rewards:0.1f}')

            self.epsilon = max(self.epsilon - self.decay_epsilon, 0)

    def stats(self, num_episodes):
        rewards_per_thousand_episodes = np.split(np.array(self.rewards_all_episodes),num_episodes/200)
        count = 200

        print("********Average reward per thousand episodes********\n")
        for r in rewards_per_thousand_episodes:
            print(count, ": ", str(sum(r/200)))
            count += 200
        #Print the updates Q-Table
        print("\n\n*******Q-Table*******\n")
        print(self.q_table)




if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    agent = Q_agent()
    agent.train_model2()
    env.close()
