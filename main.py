from torch import multiprocessing as mp

from train.Coach import Coach
from network.NNetWrapper import NNetWrapper as nn
from utils import *
from libcpp import Game

args = dotdict({
    'run_name': 'Gomoku',
    'workers': mp.cpu_count() - 1,
    'start_iter': 1,
    'num_iters': 50,
    'train_batch_size': 512,
    'train_steps_per_iteration': 200,
    'max_sample_num': 10000, 
    'num_iters_for_train_examples_history': 100,
    'temp_threshold': 10,
    'temp': 1,
    'arena_compare_random': 50,
    'arena_compare': 50,
    'arena_temp': 0.1,
    'compare_with_random': True,
    'random_compare_freq': 10,
    'compare_with_past': True,
    'past_compare_freq': 10,
    'checkpoint': 'checkpoint',
    'data': 'data',
})

if __name__ == "__main__":
    # Create a Gomoku 9x9 game instance
    g = Game(9, 5)

    # Create a neural network instance
    nnet = nn(g)

    c = Coach(g, nnet, args)
    c.learn()
