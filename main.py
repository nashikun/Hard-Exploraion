import sys
import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import dtype
from torchvision import transforms, datasets

from SAC.soft_actor_critic import SoftActorCritic
from utils.helper import moving_average


def init_config(args):
    """
    Reads the configuration from the path while overwriting any prmeters by those given in the config
    """

    with open(args.config, 'r') as f:
        config = json.load(f)
    for k, v in vars(args).items():
        if k != "config":
            config[k] = v
    return defaultdict(lambda: None, config)

def init_parser():
    """
    Initializes the parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, help="The config file path")
    parser.add_argument("--env", type=str, required=False, default="Pendulum-v0", help="The environment name")
    parser.add_argument("--seed", type=int, required=False, default=0, help="The random state seed. enter 0 for random")
    parser.add_argument("--bach_size", type=int, required=False, default=64, help="The batch size")
    parser.add_argument("--epochs", type=int, required=False, default=100, help="The number of training epochs")
    parser.add_argument("--beta", type=float, required=False, default=0.05, help="Temperature")
    parser.add_argument("--learning_rate", type=float, required=False, default=1e3, help="The learning rate")
    parser.add_argument("--cuda", dest="use_cuda", action='store_true', help="To use Cuda")
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false', help="To disable Cuda")
    parser.set_defaults(use_cuda=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = init_parser()
    config = init_config(args)
    device = torch.device('cuda' if config["use_cuda"] and torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    algo = SoftActorCritic(env=config["env"], beta=config["beta"], is_frame=False, memory_size=1000,learning_rate=config["learning_rate"])
    s = time.time()
    algo.run(epochs=config["epochs"], batch_size=config["batch_size"])

    print(f"Algorithm took {time.time() - s} seconds.")

    plt.plot(np.array(algo.rewards_summary))
    plt.plot(moving_average(algo.rewards_summary, 100))
    plt.show()


    #       ********************************
    #       *                              *
    #       *           Evaluation         *
    #       *                              *
    #       ********************************
    algo.evaluate(500)
    plt.plot(np.array(algo.eval_rewards_summary))
    plt.plot(moving_average(algo.eval_rewards_summary, 25))
    plt.show()

    # Logging
    print(f"Total number of steps {algo.steps}")
    print(f"Mean of reward is {torch.mean(torch.Tensor(algo.eval_rewards_summary))}")
