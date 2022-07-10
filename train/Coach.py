import torch
from glob import glob
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import numpy as np
import os
from tensorboardX import SummaryWriter

from train.Arena import Arena
from game.Players import RandomPlayer, NNPlayer

class Coach:
    def __init__(self, game, nnet, args):
        np.random.seed()

        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args

        networks = sorted(glob(self.args.checkpoint+'/*'))
        self.args.start_iter = len(networks)
        if self.args.start_iter == 0:
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename='iteration-0000.pkl')
            self.args.start_iter = 1

        self.nnet.load_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{(self.args.start_iter-1):04d}.pkl')

        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/'+self.args.run_name)
        else:
            self.writer = SummaryWriter()

    def learn(self):
        for i in range(self.args.start_iter, self.args.num_iters + 1):
            print(f'------ITER {i}------')
            self.train(i)
            if self.args.compare_with_random and i % self.args.random_compare_freq == 0:
                self.compareToRandom(i)
            if self.args.compare_with_past and i % self.args.past_compare_freq == 0:
                self.compareToPast(i)
            print()
        self.writer.close()

    def train(self, iteration):
        datasets = []
        currentHistorySize = min(max(4, (iteration + 4)//2),self.args.num_iters_for_train_examples_history)
        for i in range(max(1, iteration - currentHistorySize), iteration + 1):
            data_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(
                data_tensor, policy_tensor, value_tensor))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                num_workers=self.args.workers, pin_memory=True)

        train_steps = min(self.args.train_steps_per_iteration, 
            2 * (iteration + 1 - max(1, iteration - currentHistorySize)) * self.args.max_sample_num // self.args.train_batch_size)
        l_pi, l_v = self.nnet.train(dataloader, train_steps)
        self.writer.add_scalar('loss/policy', l_pi, iteration)
        self.writer.add_scalar('loss/value', l_v, iteration)
        self.writer.add_scalar('loss/total', l_pi + l_v, iteration)

        self.nnet.save_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets

    def compareToPast(self, iteration):
        past = max(0, iteration - 10)
        self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                  filename=f'iteration-{past:04d}.pkl')
        print(f'PITTING AGAINST ITERATION {past}')
        pplayer = NNPlayer(self.game, self.pnet, self.args.arena_temp)
        nplayer = NNPlayer(self.game, self.nnet, self.args.arena_temp)

        arena = Arena(nplayer.play, pplayer.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arena_compare)

        print(f'NEW/PAST WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/to past', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToRandom(self, iteration):
        r = RandomPlayer(self.game)
        nnplayer = NNPlayer(self.game, self.nnet, self.args.arena_temp)
        print('PITTING AGAINST RANDOM')

        arena = Arena(nnplayer.play, r.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arena_compare_random)

        print(f'NEW/RANDOM WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/to random', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)