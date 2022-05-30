import random
import sys
from collections import deque

import numpy as np
import tensorflow
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from Environment import Env

EPISODES = 5000
STATE_SZIE = 225

class DQNAgent:
    def __init__(self):
        # initialize values
        self.render = False
        self.load = True
        self.save_loc = './DQN'
        self.action_size = 4
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 1e-6
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.train_start = 1000
        self.memory = deque(maxlen=2000)
        self.state_size = STATE_SZIE
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        # Load Model if self.load = True
        if self.load:
            self.load_model()

    def build_model(self):
        # Neural Network for Deep Q-learning

        model = Sequential([
            Dense(256, activation='relu', input_dim=self.state_size),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation=None)
        ])
        model.summary()
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def select_action(self, state):
        # select action using epsilon-greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            # select maximum value position in predict
            pred = self.model.predict(state)
            return np.argmax(pred)

    def MEMORY(self, state, action, reward, next_state, goal, wumpus):
        # Save Values in Memory Agent
        self.memory.append((state, action, reward, next_state, goal, wumpus))

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def update_target_model(self):
        # save weights model in target model
        self.target_model.set_weights(self.model.get_weights())

    def train_replay(self):
        # train model base size min values in batch size or memory

        if len(self.memory) < self.train_start:
            return

        # split memory data to size batch_size
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        # generate array to size (batch_size, self.state_size) and filled with zeros
        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))

        action, reward, goal, wumpus = [], [], [], []

        # for in sample memory
        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            goal.append(mini_batch[i][4])
            wumpus.append(mini_batch[i][5])

        # predict models
        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        # update target model
        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if goal[i]:
                target[i][action[i]] = reward[i]
            elif wumpus[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1)

    # save the model which is under training
    def save_model(self):
        print("Model saved !")
        self.model.save('model.h5')
        self.model.save_weights('weights_model.h5')

    # load the saved model
    def load_model(self):
        try:
            self.model = tensorflow.keras.models.load_model('model.h5')
            self.model.load_weights('weights_model.h5')
            print("Model loaded !")
        except:
            print("Model not Exist")


if __name__ == "__main__":
    # create environment
    env = Env()
    agent = DQNAgent()

    scores, episodes = [], []

    # for in EPISODES
    for e in range(EPISODES):
        state = env.reset()
        goal = False
        wumpus = False
        score = 0
        steps = 0

        state = np.reshape(state, [1, STATE_SZIE])

        # check until goal or wumpus equal true
        while (not goal) and (not wumpus):
            if agent.render:
                env.render()

            steps = steps + 1

            # get action
            action = agent.select_action(state)

            # get next state and reshape
            next_state, reward, goal, wumpus = env.step(action)
            next_state = np.reshape(next_state, [1, STATE_SZIE])

            # check wumpus reward for balance rewards
            reward = reward if not wumpus or score > 499 else -100

            # save the sample <s, a, r, s'> to the memory
            agent.MEMORY(state, action, reward, next_state, goal, wumpus)

            # every time step equal train_start do the training
            if steps >= agent.train_start:
                agent.train_replay()

            score += reward
            state = next_state

            if goal:
                # update the target model weights
                agent.update_target_model()

                # add score in scores and episode in episodes
                scores.append(score)
                episodes.append(e)

                env.reset()

                print("episode: {:3}   score: {:8.6}    epsilon {:.3}"
                      .format(e, float(score), float(agent.epsilon)))
                print("steps: " + str(steps))

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

            elif wumpus:
                # update the target model weights
                agent.update_target_model()

                # add score in scores and episode in episodes
                scores.append(score)
                episodes.append(e)

                env.reset()

                print("episode: {:3}   score: {:8.6}    epsilon {:.3}"
                      .format(e, float(score), float(agent.epsilon)))
                print("steps: " + str(steps))

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model every 10 episodes
        if e % 10 == 0:
            if e != 0:
                agent.save_model()
