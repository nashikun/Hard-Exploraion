import sys
import os
import torch
from torch import dtype
import numpy as np
from SAC.soft_actor_critic import SoftActorCritic
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helper import moving_average
from torchvision import transforms, datasets
import time


#       ********************************
#       *                              *
#       *        Hyperparameters       *
#       *                              *
#       ********************************
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
BETA = 0.05
SEED = 54
dtype = float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


#       ********************************
#       *                              *
#       *           Training           *
#       *                              *
#       ********************************
algo = SoftActorCritic(env="Pendulum-v0", beta=BETA, is_frame=False, memory_size=1000,learning_rate=LEARNING_RATE)
s = time.time()
algo.run(epochs=100, batch_size=BATCH_SIZE)

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

