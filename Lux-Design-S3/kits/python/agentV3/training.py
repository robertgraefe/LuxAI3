import json
import math
import random
from collections import deque

import jax
import numpy as np
from luxai_s3.state import EnvObs, EnvState
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from typing import TypedDict, Any
import torch.nn.functional as F
from torch import nn, Tensor, optim
from agentV1_utils.environment import Observation
from agentV1 import Agent
import torch
from matplotlib import pyplot as plt
env = LuxAIS3GymEnv()
env = RecordEpisode(env)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.double()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DQN(5, 5)
target_net = DQN(5, 5)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True)

def manhatten_distance(point_a: tuple[int, int], point_b: tuple[int, int]):
    ax, ay = point_a
    bx, by = point_b
    return abs(ax - bx) + abs(ay - by)

def anticipate_reward() -> np.ndarray:
    obs_0 = before_observation["player_0"]

    destination = (0,23)

    reward_map = np.zeros((24,24))

    for x in range(24):
        for y in range(24):
            tile_type: jax.Array = obs_0.map_features.tile_type[x,y]
            tile_type: int = tile_type.item()
            if tile_type == 2:
                reward_map[x,y] = -np.inf
                continue
            reward_map[x,y] = (23-manhatten_distance((x,y),destination))/23

    return reward_map

def get_reward(before_observation: TypedDict('Observation', {'player_0': EnvObs, "player_1": EnvObs})) -> TypedDict('Reward', {'player_0': Tensor, "player_1": Tensor}):

    obs_0 = before_observation["player_0"]

    destination = (0,23)

    reward_map = np.zeros((24,24))

    for x in range(24):
        for y in range(24):
            tile_type: jax.Array = obs_0.map_features.tile_type[x,y]
            tile_type: int = tile_type.item()
            if tile_type == 2:
                reward_map[x,y] = -np.inf
                continue
            reward_map[x,y] = (23-manhatten_distance((x,y),destination))/23

    reward_0 = 0

    for unit_id in np.where(obs_0.units_mask)[0]:
        position = np.array(obs_0.units.position[0])[unit_id]
        reward_0 += reward_map[position[0], position[1]]

    return {
        "player_0": torch.as_tensor(reward_0, dtype=torch.float32),
        "player_1": torch.as_tensor(0, dtype=torch.float32)
    }

def select_action(state, step):
    sample = random.random()
    eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1. * step / 1000)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            a: Tensor = policy_net(state)
            r = a.topk(1).indices
            return r
    else:
        return torch.tensor(random.choice([0,1,2,3,4]))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append((state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def optimize_model():
    if len(memory) < 128:
        return
    batch = memory.sample(128)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.

    states = []
    actions = []
    next_states = []
    rewards = []
    for state, action, next_state, reward in batch:
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          next_states)), dtype=torch.bool).reshape(-1,1)
    non_final_next_states = torch.stack([s for s in next_states
                                                if s is not None])
    state_batch = torch.stack(states)
    action_batch = torch.as_tensor(actions).reshape(-1,1)
    reward_batch = torch.as_tensor(rewards).reshape(-1,1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(0, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(128,dtype=torch.double).reshape(-1,1)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * 0.99) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def plot_training(reward_history):
    # Calculate the Simple Moving Average (SMA) with a window size of 50
    sma = np.convolve(reward_history, np.ones(50) / 50, mode='valid')

    # Clip max (high) values for better plot analysis
    _reward_history = np.clip(reward_history, a_min=None, a_max=100)
    sma = np.clip(sma, a_min=None, a_max=100)

    plt.figure(1)
    plt.clf()
    plt.title("Obtained Rewards")
    plt.plot(_reward_history, label='Raw Reward', color='#4BA754', alpha=1)
    plt.plot(sma, label='SMA 50', color='#F08100')
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()

    plt.tight_layout()
    plt.grid(True)
    plt.pause(0.001)

    save_chart("Reward", "Roll Reward", "Step", _reward_history.tolist(), sma.tolist(), [i for i in range(_reward_history.__len__())])

def save_chart(x1_label: str, x2_label: str, y_label: str, x1: list, x2: list, y: list):
    with open('/home/robert/PycharmProjects/LuxAI3/GUI/src/assets/data.json', 'w', encoding='utf-8') as file:
        dump = {
            "x1_label": x1_label,
            "x2_label": x2_label,
            "y_label": y_label,
            "x1": x1,
            "x2": x2,
            "y": y
        }
        json.dump(dump, file)


""" Training """
observation, info = env.reset()
observation: TypedDict('Observation', {'player_0': EnvObs, "player_1": EnvObs}) = observation
info: TypedDict('Info', {'params': dict[str, int], "full_params": dict[str, Any], "state": EnvState}) = info
done = False
truncation = False
step_size = 0
episode_reward = 0
agent_0 = Agent("player_0", info["params"])
agent_1 = Agent("player_1", info["params"])
step = 0
memory = ReplayMemory(10000)
reward_history = []

while not done and not truncation:
    print(step)
    actions = dict()

    before_observation: TypedDict('Observation', {'player_0': EnvObs, "player_1": EnvObs}) = observation

    actions_0 = np.zeros((16,3), dtype=int)
    obs_0 = Observation(0,1,before_observation["player_0"], actions_0)

    action = None
    state = None

    if obs_0.player.units:

        for unit in obs_0.player.units.values():
            reward_map = anticipate_reward()
            x = unit.position[0]
            y = unit.position[1]
            reward_center = reward_map[x, y]
            reward_up = reward_map[x, y - 1] if y - 1 >= 0 else -np.inf
            reward_right = reward_map[x + 1, y] if x + 1 <= 23 else -np.inf
            reward_down = reward_map[x, y + 1] if y + 1 <= 23 else -np.inf
            reward_left = reward_map[x - 1, y] if x - 1 >= 0 else -np.inf
            state = torch.as_tensor([reward_center, reward_up, reward_right, reward_down, reward_left])
            action = select_action(state, step)
            unit.move_direction(action.item())

    actions[agent_0.player] = actions_0
    actions[agent_1.player] = agent_1.act(step, observation["player_1"])

    observation, _, terminated, truncated, info = env.step(actions)
    reward = get_reward(observation)["player_0"]

    obs_0 = Observation(0,1,before_observation["player_0"], actions_0)
    if obs_0.player.units:

        for unit in obs_0.player.units.values():
            reward_map = anticipate_reward()
            x = unit.position[0]
            y = unit.position[1]
            reward_center = reward_map[x, y]
            reward_up = reward_map[x, y - 1] if y - 1 >= 0 else -np.inf
            reward_right = reward_map[x + 1, y] if x + 1 <= 23 else -np.inf
            reward_down = reward_map[x, y + 1] if y + 1 <= 23 else -np.inf
            reward_left = reward_map[x - 1, y] if x - 1 >= 0 else -np.inf
            next_state = torch.as_tensor([reward_center, reward_up, reward_right, reward_down, reward_left])

            if state is not None and action is not None:
                memory.push(state, action, next_state, reward)

            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (
                            1 - 0.005)
            target_net.load_state_dict(target_net_state_dict)

            terminated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})
            truncated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})

            if terminated["player_0"].item() or terminated["player_1"].item() or truncated["player_0"].item() or truncated["player_1"].item():
                done = True
                truncation = True

    step += 1
    reward_history.append(reward)
    plot_training(reward_history)

env.save_episode("test.json")
#env.close()