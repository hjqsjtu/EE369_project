# -*-coding:utf-8-*-
import numpy as np
import config as conf
import numpy.lib.format as fmt

import random
import tflearn
import copy
import sys
import tensorflow as tf

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


from game2048.game import Game


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


class NNAgent(Agent):
    def __init__(self, game, display=None):
        self.game = game
        self.display = display
        if game.size != 4:
            raise ValueError("`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game,display)

        self.model = None
        self.network = None
        self.modelisloaded = False
        # model has been trained ok to be used in step()
        self.trainedok = False
        # trained and save model into a existed file
        self.train_continue = True
        # load sample data from file
        self.traindata_canload = False
        # sample been there
        self.sample_exist = True
        self.training_data = None
        self.scores = []
        tf.reset_default_graph()
        if self.trainedok:
            self.create_network(16)
            # self.model.reset_default_graph()
            try:
                self.model.load('{}.model'.format('SJTU_HJQ2048'),weights_only=True)
                self.modelisloaded = True
            except Exception as e:
                print('Model not found')
                self.modelisloaded = False

    def train(self):
        filename = 'expect_move'
        if self.traindata_canload:
            self.training_data = np.load('{}.npy'.format(filename)).tolist()
        else:
            # self.init_gens('expect_move')
            self.gen_sample(filename)
        self.train_model()
        self.model.save('{}.model'.format('SJTU_HJQ2048'))

    def step(self):
        if not self.modelisloaded:
            return -1
        obs_his = None
        obs_his = np.concatenate(self.game.board, axis=0)
        prediction = sorted_prediction(self.model.predict(obs_his.reshape(-1, len(obs_his),1))[0])
        tmpgame = copy.deepcopy(self.game)
        done = False
        for i in range(len(prediction)):
            action = conf.options[prediction[i]]
            tmpgame.move(action)
            if not equals_grid(np.concatenate(tmpgame.board, axis=0), obs_his):
                done = True
                break
        if done:
            return action
        else:
            return -1

    def gen_sample(self, heuristic='explorer'):
        self.training_data = []
        #
        self.scores = []
        self.accepted_scores = []
        continued = -1
        i = 0
        j = 0
        k = 0
        print('Starting to generating data ...')
        try:
            while i < conf.initial_games or len(self.accepted_scores) < int(conf.initial_games * 0.2):
                # gamegrid = inv_puzzle.Game()
                score = 0
                game_memory = []
                prev_observation = []
                choices = []
                localagent = None
                if heuristic == 'expect_move':
                    localagent = ExpectiMaxAgent(self.game)
                k = 0
                for k in range(conf.goal_steps):
                    prev_observation = np.concatenate(self.game.board, axis=0)
                    invalid_moves = []
                    while equals_grid(np.concatenate(self.game.board, axis=0), prev_observation):
                        if heuristic == 'random':
                            move = [x for x in conf.options if x not in invalid_moves]
                            action = random.choice(move)
                        elif heuristic == 'expect_move':
                            action = localagent.step()
                        elif heuristic == 'corner':
                            action = corner_choice(self.game.board, conf.options, invalid_moves)
                        elif heuristic == 'one_corner':
                            action = left_down_corner_choice(self.game.board, conf.options, invalid_moves)
                        elif heuristic == 'explorer':
                            action, _ = explorer_move(self.game, conf.options, 3)  # Max depth == 3
                            if action in invalid_moves:  # Just in case i   t fail
                                action = random.choice([x for x in conf.options if x not in invalid_moves])
                        invalid_moves.append(action)
                        self.game.move(action)
                        continued = self.game.end
                        if continued == 1:
                            done = True
                        else:
                            done = False
                        if done:
                            break
                    choices.append(action)
                    game_memory.append([prev_observation, action])
                    # score += self.game.score - score  # How much we win with this move
                    score = self.game.score
                    save_score = score >= conf.score_requirement
                    k += 1
                    if done or save_score:
                        break

                save_score = score >= conf.score_requirement
                             # and self.reach2048()
                if save_score:
                    self.accepted_scores.append(score)
                    j += 1
                    for data in game_memory:
                        output = [0, 0, 0, 0]
                        output[conf.options.index(data[1])] = 1
                        self.training_data.append([data[0], output])

                self.scores.append(score)
                i += 1
                print('Game: {}, Passed: {}/{}, Steps:{},Score: {}, Saved: {}'.format(i, len(self.accepted_scores),
                                                                             i, k, score, save_score))
                print('a: {}%, s: {}%, d: {}%, w: {}%'.format(
                    round(choices.count(conf.options[0]) / len(choices) * 100, 2),
                    round(choices.count(conf.options[1]) / len(choices) * 100, 2),
                    round(choices.count(conf.options[2]) / len(choices) * 100, 2),
                    round(choices.count(conf.options[3]) / len(choices) * 100, 2)
                ))
                self.game = Game(4, 2048)
        except KeyboardInterrupt as e:
            print('Training stoped')
            if i > 0:
                file = '{}.npy'.format(heuristic)
                tmptraining_data = None
                if self.sample_exist:
                    tmptraining_data = np.load(file).tolist()
                    tmptraining_data += self.training_data
                else:
                    tmptraining_data = self.training_data
                training_data_save = np.array(tmptraining_data)
                np.save(file, training_data_save)
            sys.exit()

        if i > 0:
            # self.training_data.append(i)
            file = '{}.npy'.format(heuristic)
            tmptraining_data = None
            if self.sample_exist:
                tmptraining_data = np.load(file).tolist()
                tmptraining_data += self.training_data
            else:
                tmptraining_data = self.training_data
            training_data_save = np.array(tmptraining_data)
            np.save(file, training_data_save)
            #fid = open(file, "a+b")
            # arr = np.asanyarray(training_data_save)
            # fmt.write_array(fid, arr)
            # fid.close()
            #np.save(fid, training_data_save)
            print('Average accepted score: ', mean(self.accepted_scores))
            print('Median accepted score: ', median(self.accepted_scores))
            print(Counter(self.accepted_scores))
            # self.training_data = self.training_data[:-1]

    def convert_sample(self):
        filename = 'expect_move'
        self.training_data = np.load('{}.npy'.format(filename)).tolist()
        samples = []
        col1 = None
        supp = 0
        for idata in self.training_data:
            idata[0] = np.log1p(idata[0])
            supp = max(idata[0])
            idata[0] = idata[0]/supp
            # col1 = np.array(idata[1])
            samples.append([idata[0], idata[1]])
        samples_save = np.array(samples)
        filename = 'expect_move_conv.npy'
        np.save(filename, samples_save)

    def train_model(self):
        y_data = [i[1] for i in self.training_data]
        x_data = np.array([i[0] for i in self.training_data]).reshape(-1, len(self.training_data[0][0]), 1)
        ###################################
        ###################################
        self.create_network(input_size=len(x_data[0]))
        if self.train_continue:
            try:
                self.model.load('{}.model'.format('SJTU_HJQ2048'), weights_only=True)
            except Exception as e:
                pass
        self.model.fit({'input': x_data}, {'targets': y_data},
                       n_epoch=10, snapshot_step=5000, show_metric=True, run_id='SJTU_HJQGAME2048')

    def create_network(self, input_size):
        self.network = input_data(shape=[None, input_size, 1], name='input')

        self.network = fully_connected(self.network, 128, activation='sigmoid')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 256, activation='sigmoid')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 512, activation='sigmoid')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 256, activation='sigmoid')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 128, activation='sigmoid')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 4, activation='softmax')
        self.network = regression(self.network, optimizer='adam', learning_rate=conf.LR, loss='categorical_crossentropy', name='targets')

        self.model = tflearn.DNN(self.network, tensorboard_dir='log')

    def reach2048(self):
        for i in range(len(self.game.board)):
            for j in range(len(self.game.board[0])):
                if self.game.board[i][j] >= 2048:
                    return True
        return False


def explorer_move(game, options, depth):
    score = 0
    move = options[0]
    gg = np.concatenate(game.board, axis=0)

    for i in range(4):
        g = copy.deepcopy(game)
        g.move(options[i])
        continued = g.end
        if continued == 0:
            done = False
        else:
            done = True

        # done = g.move(options[i])
        own_s = g.score - game.score
        # Our base case is when depth reached (0), finish the game or invalid move (the last one is just for truncate)
        if depth > 0 and not done and not equals_grid(gg, np.concatenate(g.board, axis=0)):
            m, s = explorer_move(g, options, depth - 1)
            own_s += s
            #own_s = s
        # Here we choose just the best option
        if own_s > score:
            score = own_s
            move = options[i]
    return move, score


def corner_choice(matrix, options, invalid_moves):
    one_line_matrix = np.concatenate(matrix, axis=0)
    index_max_value = one_line_matrix.argmax(axis=0)

    line = index_max_value / len(matrix[0])
    column = index_max_value % len(matrix[0])

    if column == 0 and options[1] not in invalid_moves:
        if random.random() < 0.5:
            return options[1] # left
    if column == len(matrix[0])-1 and options[3] not in invalid_moves:
        if random.random() < 0.5:
            return options[3] # right
    if line == 0 and options[0] not in invalid_moves:
        if random.random() < 0.5:
            return options[0] # up
    if line == len(matrix[0]) and options[2] not in invalid_moves:
        if random.random() < 0.5:
            return options[2] # down

    move = [x for x in options if x not in invalid_moves]
    action = random.choice(move)
    return action


def left_down_corner_choice(matrix, options, invalid_moves):
    one_line_matrix = np.concatenate(matrix, axis=0)
    index_max_value = one_line_matrix.argmax(axis=0)

    line = index_max_value / len(matrix[0])
    column = index_max_value % len(matrix[0])

    if (column == 0 or random.random() < 0.5) and options[1] not in invalid_moves:
        return options[1] # left
    if (line == len(matrix[0]) or random.random() < 0.5) and options[2] not in invalid_moves:
        return options[2] # down
    move = [x for x in options if x not in invalid_moves]
    action = random.choice(move)
    return action


def equals_grid(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def sorted_prediction(prediction):
    return sorted(range(len(prediction)), key=lambda k: prediction[k], reverse=True)

