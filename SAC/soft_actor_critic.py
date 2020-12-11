# from memory.replay import ReplayBuffer
import gym
from utils.processor import *
from .models import *
from utils.memory import *
from torch import optim
from torch.autograd import Variable
from utils.memory import ReplayBuffer
import torch
import torch.distributions as D
import time
from utils.activation import *

CONFIG = {
    "gamma": 0.99,
    "memory_size": 1000,
    "learning_rate": 0.0002,
    "environment": "Pong-v0",
    "device": "cpu",
    "update_step": 10000,
    "training_step": 4,
    "tau": 0.005
}


class SoftActorCritic:
    def __init__(self,
                 is_frame=True,
                 gamma=0.99,
                 beta=0.05,
                 tau=0.005,
                 learning_rate=0.0002,
                 memory_size=64,
                 env="Pong-v0",
                 device="cpu",
                 seed=42):

        self.device = device
        # Initialize the environment
        self.env = gym.make(env)
        self.action_range = [self.env.action_space.low, self.env.action_space.high]
        print(self.action_range)
        print(self.env.action_space)
        self.env.reset()
        if seed:
            torch.manual_seed(seed)
            self.env.seed(seed)

        self.num_actions = self.env.action_space.shape[0]
        self.input_dim = self.env.observation_space.shape[0]
        self.is_frame = is_frame
        # Initialize the actor and the critic
        self.actor = PolicyNetwork(num_inputs=self.input_dim, num_actions=self.num_actions)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.soft_q_1 = SoftQNetwork(num_inputs=self.input_dim, num_actions=self.num_actions)
        self.soft_q_1_optimizer = optim.Adam(self.soft_q_1.parameters(), lr=learning_rate)

        self.soft_q_2 = SoftQNetwork(num_inputs=self.input_dim, num_actions=self.num_actions)
        self.soft_q_2_optimizer = optim.Adam(self.soft_q_2.parameters(), lr=learning_rate)

        self.value_network = ValueNetwork(input_dim=self.input_dim, output_dim=1)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)

        self.target_value_network = ValueNetwork(input_dim=self.input_dim, output_dim=1)

        for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(param.data)

        self.loss = nn.SmoothL1Loss()
        # Initialize other parameters
        self.memory = ReplayBuffer(memory_size)
        self.gamma = gamma
        self.beta = beta
        self.tau = tau
        # Logging
        self.rewards_summary = []
        self.eval_rewards_summary = []
        self.history = []
        self.steps = 0
        self.update_step = 0
        self.delay_step = 2
        self.loss = nn.MSELoss()

    def act(self, s) -> tuple:
        input_state = s.reshape(1, -1)
        mean, log_std = self.actor(input_state)
        std = log_std.exp()

        normal = D.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + (
                    self.action_range[1] + self.action_range[0]) / 2.0

    def compute_discounted_rewards(self, rewards):
        discounted_rewards = np.array([self.gamma ** i * rewards[i] for i in range(len(rewards))])
        discounted_rewards = discounted_rewards[::-1].cumsum()[::-1]
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (
                np.std(discounted_rewards) + 1e-5)
        return discounted_rewards

    def compute_loss(self, rewards, logprob, entropy):
        discounted_rewards = self.compute_discounted_rewards(rewards)
        error = -np.dot(discounted_rewards, logprob) + self.beta * entropy
        return error

    def update(self, s_batch, a_batch, r_batch, d_batch, s2_batch) -> float:

        a2_batch, next_log_pi = self.actor.sample(s2_batch)

        next_q_1 = self.soft_q_1(s2_batch, a2_batch).squeeze()  # First Q Network
        next_q_2 = self.soft_q_2(s2_batch, a2_batch).squeeze()  # Second Q network

        predicted_q_1 = self.soft_q_1(s_batch, a_batch).squeeze()  # First Q Network
        predicted_q_2 = self.soft_q_2(s_batch, a_batch).squeeze()  # Second Q network

        target_value = self.target_value_network(s2_batch).squeeze()  # Value network
        target_q_value = r_batch + (1 - d_batch) * self.gamma * target_value

        q_1_loss = self.loss(predicted_q_1, target_q_value)
        q_2_loss = self.loss(predicted_q_2, target_q_value)

        next_v_target = torch.min(next_q_1, next_q_2) - next_log_pi.squeeze()
        curr_v = self.value_network.forward(s_batch).squeeze()
        v_loss = self.loss(curr_v, next_v_target.detach())
    
        # update value network and q networks
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()

        self.soft_q_1_optimizer.zero_grad()
        q_1_loss.backward(retain_graph=True)
        self.soft_q_1_optimizer.step()

        self.soft_q_2_optimizer.zero_grad()
        q_2_loss.backward()
        self.soft_q_2_optimizer.step()
        # delayed update for policy net and target value nets
        if not self.update_step % self.delay_step:
            new_actions, log_pi = self.actor.sample(s_batch)
            min_q = torch.min(
                self.soft_q_1.forward(s_batch, a2_batch),
                self.soft_q_2.forward(s_batch, a2_batch)
            )
            policy_loss = (log_pi - min_q).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # target networks
            for target_param, param in zip(self.target_value_network.parameters(), self.value_network.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        self.update_step += 1

    def evaluate(self, episodes=100):
        for i in range(episodes):
            # Initialize sequence and preprocess it
            print(f"Episode #{i}")
            self.env.reset()
            is_done = False

            if self.is_frame:
                frames = []
                for _ in range(4):
                    frame, _, is_done, _ = self.env.step(0)
                    frames.append(frame)

                s = make_state(frames)
            else:
                frame, _, is_done, _ = self.env.step([0])
                s = torch.from_numpy(frame).float()
            N = 0
            episode_reward = 0

            while not is_done and N < 500:
                action = self.act(s)
                s, reward, is_done, _ = self.env.step(action)
                s = torch.from_numpy(s).float()
                episode_reward += reward
                N += 1

            self.eval_rewards_summary.append(episode_reward)

    def run(self, epochs=10, batch_size=32) -> None:
        for j in range(epochs):
            for i in range(batch_size):
                print(f"Episode #{j * batch_size + i}")
                self.env.reset()
                is_done = False

                if self.is_frame:
                    frames = []
                    for _ in range(4):
                        frame, _, is_done, _ = self.env.step(0)
                        frames.append(frame)

                    s = make_state(frames)
                else:
                    frame, _, is_done, _ = self.env.step([0])
                    s = torch.from_numpy(frame).float()
                mean_error = 0
                N = 0
                episode_reward = 0
                rewards = []
                while not is_done and N < 500:
                    action = self.act(s)
                    # Execute action a_t in emulator and observe reward r_t and image x_(t+1)
                    frame, reward, is_done, _ = self.env.step(action)

                    if N >= 500:
                        is_done = True
                    episode_reward += reward
                    rewards.append(reward)
                    # Set s_(t+1) = s_t, a_t, x_(t+1) and preprocess φ_(t+1) = φ(s_(t+1))
                    if self.is_frame:
                        s_next = torch.from_numpy(
                            np.append(s[1:, :, :], process_frame(frame).reshape((1, 84, 84)), axis=0))
                    else:
                        s_next = torch.from_numpy(frame).float()

                    self.memory.store(s, action, reward, is_done, s_next)
                    s = s_next
                    self.steps += 1
                    N += 1
                self.rewards_summary.append(episode_reward)

            s_batch, a_batch, r_batch, d_batch, s2_batch = self.memory.sample(batch_size, True)

            self.update(s_batch, a_batch, r_batch, d_batch, s2_batch)
