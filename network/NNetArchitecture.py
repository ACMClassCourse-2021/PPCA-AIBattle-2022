import torch.nn.functional as F
import torch.nn as nn
import torch
import sys

sys.path.append('..')

class NNetArchitecture(nn.Module):
    def __init__(self, game, args):
        super(NNetArchitecture, self).__init__()
        # game params
        self.feat_cnt = args.feat_cnt
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        """
            TODO: Add anything you need
        """

    def forward(self, s):
        # batch_size x feat_cnt x board_x x board_y
        s = s.view(-1, self.feat_cnt, self.board_x, self.board_y)   

        """
            TODO: Design your neural network architecture
            Return a probability distribution of the next play (an array of length self.action_size) 
            and the evaluation of the current state.

            pi = ...
            v = ...
        """

        # Think: What are the advantages of using log_softmax ?
        return F.log_softmax(pi, dim=1), torch.tanh(v)