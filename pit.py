from utils import *

from train.Arena import Arena
from game.Players import *
from network.NNetWrapper import NNetWrapper as NNet
from libcpp import Game

if __name__ == '__main__':

    g = Game(9, 5)

    # all players
    # rp = RandomPlayer(g).play
    # hp = HumanPlayer(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./checkpoint/', 'iteration-0050.pkl')
    p1 = NNPlayer(g, n1, 0).play

    n2 = NNet(g)
    n2.load_checkpoint('./checkpoint/', 'iteration-0025.pkl')
    p2 = NNPlayer(g, n2, 0).play

    arena = Arena(p1, p2, g, display=g.display)
    print(arena.playGames(2, verbose=True))
