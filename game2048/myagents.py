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
from tflearn.optimizers import adam
from statistics import mean, median
from collections import Counter
from tflearn.callbacks import Callback
import statistics

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


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        # Store a validation accuracy threshold, which we can compare against
        # the current validation accuracy at, say, each epoch, each batch step, etc.
        self.val_acc_thresh = val_acc_thresh
        self.accs = []

    def on_epoch_end(self, training_state):
        self.accs.append(training_state.global_acc)
        if training_state.val_acc is not None and training_state.val_acc > self.val_acc_thresh:
            raise StopIteration


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
        self.traindata_canload = True
        # sample been there
        self.sample_exist = True
        self.training_data = []
        self.scores = []
        tf.reset_default_graph()
        if self.trainedok:
            self.create_network(20)
            # self.model.reset_default_graph()
            try:
                self.model.load('{}.model'.format('NN_CUSTOM_2048'),weights_only=True)
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
        self.model.save('{}.model'.format('NN_CUSTOM_2048'))

    def step(self):
        if not self.modelisloaded:
            return -1
        obs_his = None
        obs_his0 = np.concatenate(self.game.board, axis=0)
        obs_his = obs_his0
        prev_args = [self.emptycelss(self.game), self.monotonicity(self.game), self.smoothness(self.game),
                     5*np.log2(self.game.score + (self.game.score == 0))]
        obs_his = np.log2(obs_his + (obs_his == 0))
        prev_observation = np.concatenate([obs_his, prev_args], axis=0)
        prev_observation = (prev_observation - mean(prev_observation)) / statistics.stdev(prev_observation)
        # prev_observation = prev_observation / max(prev_observation)
        prediction = sorted_prediction(self.model.predict(prev_observation.reshape(-1, len(prev_observation),1))[0])
        tmpgame = copy.deepcopy(self.game)
        done = False
        for i in range(len(prediction)):
            action = conf.options[prediction[i]]
            tmpgame.move(action)
            if not equals_grid(np.concatenate(tmpgame.board, axis=0), obs_his0):
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
                score = 0
                action = 0
                game_memory = []
                prev_observation = []
                prev_args = []
                choices = []
                localagent = None
                if heuristic == 'expect_move':
                    localagent = ExpectiMaxAgent(self.game)
                k = 0
                for k in range(conf.goal_steps):
                    prev_observation = np.concatenate(self.game.board, axis=0)
                    prev_args = [self.emptycelss(self.game), self.monotonicity(self.game), self.smoothness(self.game), 5*np.log2(self.game.score + (self.game.score == 0)) ]
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
                    prev_observation = np.log2(prev_observation + (prev_observation == 0))
                    prev_observation = np.concatenate([prev_observation, prev_args], axis=0)
                    prev_observation = (prev_observation - mean(prev_observation)) / statistics.stdev(prev_observation)
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
                self.game = Game(4, conf.score_requirement)
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
        for idata in self.training_data:
            idata[0] = (idata[0] - statistics.mean(idata[0])) / statistics.stdev(idata[0])
            samples.append([idata[0], idata[1]])
        samples_save = np.array(samples)
        filename = 'expect_move_part.npy'
        np.save(filename, samples_save)

    """
    def convert_sample(self):
        filename = 'expect_move'
        self.training_data = np.load('{}.npy'.format(filename)).tolist()
        samples = []
        col1 = None
        supp = 0
        slen = len(self.training_data) / 10
        ilen = 0
        for idata in self.training_data:
            # supp = max(idata[0])
            # idata[0] = idata[0]/supp
            samples.append([idata[0], idata[1]])
            ilen += 1
            if ilen > slen:
                break
        samples_save = np.array(samples)
        filename = 'expect_move_part.npy'
        np.save(filename, samples_save)
    """

    def train_model(self):
        y_data = [i[1] for i in self.training_data]
        x_data = np.array([i[0] for i in self.training_data]).reshape(-1, len(self.training_data[0][0]), 1)
        ###################################
        ###################################
        self.create_network(input_size=len(x_data[0]))
        if self.train_continue:
            try:
                # self.model.load('{}.model'.format('NN_CUSTOM_2048'), weights_only=True)
                self.model.load('{}.model'.format('NN_CUSTOM_2048'))
            except Exception as e:
                pass
        # validation_set=0.05,All_data*0.05 be validation data
        # batch_size=64,update network per 64 data
        # n_epoch=50, train 50 iteration
        # shuffle=True, refresh train status info
        early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.8)
        try:
            # self.model.fit({'input': x_data}, {'targets': y_data},
            #                n_epoch=50, snapshot_step=1000, validation_set=0.05,
            #                shuffle=True, batch_size=64, run_id='SJTU_HJQGAME2048')
            self.model.fit({'input': x_data}, {'targets': y_data},
                           n_epoch=800, snapshot_step=500, validation_set=0.15,
                           shuffle=True, batch_size=32, show_metric=True, run_id='SJTU_HJQGAME2048',
                           callbacks=early_stopping_cb)
        except StopIteration as e:
            print("Left training")

    def monotonicity(self, ggame):
        # calculate monotonicity of current game.board
        totals =[0, 0, 0, 0]
        # up / down direction
        for x in range(4):
            current = 0
            next = current+1
            while next < 4:
                while next < 4 and ggame.board[x,next] == 0:
                    next += 1
                if next >= 4:
                    next -= 1
                currentvalue = np.log10(ggame.board[x, current] +(ggame.board[x, current] == 0))
                nextvalue = np.log10(ggame.board[x, next] +(ggame.board[x, next] == 0))
                if currentvalue > nextvalue:
                    totals[0] += nextvalue - currentvalue
                elif nextvalue > currentvalue:
                    totals[1] += currentvalue - nextvalue
                current = next
                next += 1
        #left / right direction
        for y in range(4):
            current = 0
            next = current+1
            while next < 4:
                while next < 4 and ggame.board[next, y] == 0:
                    next += 1
                if next >= 4: next -= 1
                currentvalue = np.log10(ggame.board[current, y] +(ggame.board[current, y] == 0))
                nextvalue = np.log10(ggame.board[next, y] +(ggame.board[next, y] == 0))
                if currentvalue > nextvalue:
                    totals[2] += nextvalue - currentvalue
                elif nextvalue > currentvalue:
                    totals[3] += currentvalue - nextvalue
                current = next
                next += 1
        return min(totals[0], totals[1]) + min(totals[2], totals[3])
        #return np.sum(totals)

    def smoothness(self, ggame):
        # calculate somoothness of current game.board
        smoothness = 0
        for x in range(4):
            for y in range(4):
                if not ggame.board[x, y] == 0:
                    value = np.log10(ggame.board[x,y])
                    mvdir = [0, 1]
                    curpos = [x, y]
                    cell = []
                    targetcell = []
                    cell, targetcell = self.findfarthestposition(ggame, curpos, mvdir)
                    if not ggame.board[targetcell[0], targetcell[1]] == 0:
                        targetvalue = np.log10(ggame.board[targetcell[0], targetcell[1]])
                        smoothness -= np.abs(value - targetvalue)
                    mvdir = [1, 0]
                    curpos = [x, y]
                    cell, targetcell = self.findfarthestposition(ggame, curpos, mvdir)
                    if not ggame.board[targetcell[0], targetcell[1]] == 0:
                        targetvalue = np.log10(ggame.board[targetcell[0], targetcell[1]])
                        smoothness -= np.abs(value - targetvalue)

        return smoothness

    def emptycelss(self, ggame):
        # calculate emptycells of game.board, greater be better
        cntemptycells = 0
        for x in range(4):
            for y in range(4):
                if ggame.board[x, y] == 0:
                    cntemptycells += 1
        return cntemptycells

    def findfarthestposition(self, ggame, cell, tcell):
        previous = cell
        cell = [previous[0] + tcell[0], previous[1] + tcell[1]]
        if cell[0] == 4 or cell[1] == 4:
            return previous, previous
        while cell[0] >= 0 and cell[0] < 3 and cell[1] >= 0 and cell[1] < 3 and ggame.board[cell[0], cell[1]] == 0:
            previous = cell
            cell = [previous[0] + tcell[0], previous[1] + tcell[1]]
        return previous, cell

    def create_network(self, input_size):
        self.network = input_data(shape=[None, input_size, 1], name='input')

        #self.network = tflearn.layers.conv_1d(self.network, 64, 3, activation='relu')
        #self.network = tflearn.layers.max_pool_1d(self.network, 2)
        #self.network = tflearn.layers.batch_normalization(self.network)
        # self.network = tflearn.layers.local_response_normalization(self.network)
        self.network = fully_connected(self.network, 64, activation='relu', weights_init='xavier')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 128,  activation='relu', weights_init='xavier')
        self.network = dropout(self.network, 0.8)

        #self.network = fully_connected(self.network, 256, activation='sigmoid', weights_init='xavier')
        #self.network = dropout(self.network, 0.8)

        #self.network = fully_connected(self.network, 512, activation='sigmoid', weights_init='xavier')
        #self.network = dropout(self.network, 0.8)

        #self.network = fully_connected(self.network, 256, activation='sigmoid', weights_init='xavier')
        #self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 128,  activation='relu', weights_init='xavier')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 64,  activation='relu', weights_init='xavier')
        self.network = dropout(self.network, 0.8)

        self.network = fully_connected(self.network, 4, activation='softmax')
        optim = adam(learning_rate=conf.LR, beta1=0.9, beta2=0.999,
                 epsilon=1e-1, use_locking=False, name="Adam")
        self.network = regression(self.network, optimizer=optim, learning_rate=conf.LR, loss='categorical_crossentropy', name='targets')

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

