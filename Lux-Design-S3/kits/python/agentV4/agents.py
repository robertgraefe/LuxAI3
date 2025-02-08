import numpy as np
import torch.nn.functional as functional
import math

from luxai_s3.state import EnvObs
from torch import nn, Tensor, optim
from torchrl.data import ReplayBuffer, LazyTensorStorage, ListStorage
import torch

# custom
from ReplayBuffer import ReplayMemory
from environment import EnvironmentConfig, Observation, TILETYPE
from actions import DIRECTION

class DQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.double()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x: Tensor):
        x = functional.relu(self.layer1(x))
        x = functional.relu(self.layer2(x))
        return self.layer3(x)

class Agent:
    def __init__(self, player: str, env_cfg):
        self.policy_net = DQN(5,5)
        self.target_net = DQN(5, 5)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=1e-4, amsgrad=True)
        self.buffer = ReplayBuffer(storage=ListStorage(10000), batch_size=128)

        self.env_cfg = EnvironmentConfig(**env_cfg)
        self.player = player
        self.opponent = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0
        self.observation: Observation

    def soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (
                    1 - 0.005)
        self.target_net.load_state_dict(target_net_state_dict)

    def act(self, step, initialObservation: EnvObs) -> dict[int, list]:

        actions = dict()

        self.observation = Observation(self.team_id, self.opponent_team_id, initialObservation, actions=np.array([]))

        for unit in self.observation.player.units.values():
            a = unit.position[0]
            b = unit.position[1]
            state = []
            # center, up, right, down, left
            for x,y in [(a,b),(a,b-1),(a+1,b),(a,b+1),(a-1,b)]:
                x = int(x)
                y = int(y)
                if x < 0 or y < 0 or x >= self.observation.map.x_max or y >= self.observation.map.y_max\
                            or ((x,y) in self.observation.map.tiles.keys() and self.observation.map.tiles[x,y].type == TILETYPE.ASTEROID):
                    state.append(torch.inf)
                    continue
                state.append(self.observation.map.tiles[x,y].index)
            state = torch.tensor(state, dtype=torch.double)

            sample = torch.rand([1])

            eps_threshold = 0.05 + (0.9 - 0.05) * math.exp(-1. * step / 1000)
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    a: Tensor = self.policy_net(state)
                    direction = a.topk(1).indices.item()
                    actions[unit.id] = [direction, 0, 0]
            else:
                # select a random action
                action_space = torch.tensor([DIRECTION.CENTER, DIRECTION.UP, DIRECTION.RIGHT, DIRECTION.DOWN, DIRECTION.LEFT])
                probabilities = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
                probabilities = probabilities.multinomial(num_samples=1, replacement=False)
                actions[unit.id] = [action_space[probabilities].item(), 0 , 0]

        return actions

    def optimize_model(self):
        if len(self.buffer) < 128:
            return
        batch = self.buffer.sample()
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
                                                next_states)), dtype=torch.bool).reshape(-1, 1)
        non_final_next_states = torch.stack([s for s in next_states
                                             if s is not None])
        state_batch = torch.stack(states)
        action_batch = torch.as_tensor(actions).reshape(-1, 1)
        reward_batch = torch.as_tensor(rewards).reshape(-1, 1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(0, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(128, dtype=torch.double).reshape(-1, 1)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * 0.99) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

class DoNothingAgent:
    def __init__(self, player: str, env_cfg) -> None:
        self.env_cfg = EnvironmentConfig(**env_cfg)
        self.player = player

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        return torch.zeros((self.env_cfg.max_units, 3), dtype=torch.int)