# -*-coding:utf-8-*-

from game2048.game import Game
from game2048.displays import Display
from game2048.myagents import NNAgent as TestAgent


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048

    game = Game(4, SCORE_TO_WIN)
    agent = TestAgent(game, display=Display())
    # agent.convert_sample()
    agent.train()
    print('Training END.')
