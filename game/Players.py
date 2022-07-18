import numpy as np

class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, turn):
        valids = self.game.getValidMoves(board, 1, turn)
        valids = valids / np.sum(valids)
        a = np.random.choice(self.game.getActionSize(), p=valids)
        return a

class NNPlayer:
    def __init__(self, game, nn, temp=1, temp_threshold=0):
        self.game = game
        self.nn = nn
        self.temp = temp
        self.temp_threshold = temp_threshold

    def play(self, board, turn):

        """
            Feature planes (one hot):
                Feat 1 (9x9 plane): own pawns
                Feat 2 (9x9 plane): opponent's pawns
                Feat 3 (9x9 plane): all 1

            TODO(optional): you can design your own feature planes
        """
        feat = np.array(board)
        feat = np.array([feat == 1, feat == -1, np.ones(feat.shape)])

        policy, _ = self.nn.predict(feat)
        valids = self.game.getValidMoves(board, 1, turn)
        options = policy * valids
        temp = 1 if turn <= self.temp_threshold else self.temp
        if temp == 0:
            bestA = np.argmax(options)
            probs = [0] * len(options)
            probs[bestA] = 1
        else:
            probs = [x ** (1. / temp) for x in options]
            probs /= np.sum(probs)

        choice = np.random.choice(
            np.arange(self.game.getActionSize()), p=probs)

        assert valids[choice] > 0

        return choice

class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board, turn):
        valid = self.game.getValidMoves(board, 1, turn)
        while True:
            a = input()

            x, y = [int(x) for x in a.split(' ')]
            a = 9 * x + y if x != -1 else 9 ** 2
            if valid[a]:
                break
            else:
                print('Invalid')

        return a