import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from typing import TypedDict
from luxai_s3.state import EnvObs
import jax
import torch
# custom
from agents import DoNothingAgent
from agents import Agent
from environment import TILETYPE, Observation
from metrics.distance import manhatten_distance
from visualisation.utils import save_chart
from actions import DIRECTION

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

agent_0 = None
agent_1 = None

reward_history = []

for episode in range(100):
    step = 0
    env = LuxAIS3GymEnv()
    env = RecordEpisode(env)
    lux_observation, info = env.reset()
    lux_observation: TypedDict('Observation', {'player_0': EnvObs, "player_1": EnvObs}) = lux_observation
    done = False
    truncation = False
    if agent_0 is None or agent_1 is None:
        agent_0 = DoNothingAgent("player_0", info["params"])
        agent_1 = Agent("player_1", info["params"])

    while not done and not truncation:
        step += 1
        actions = dict()

        actions[agent_0.player]: np.ndarray = np.array(agent_0.act(step, lux_observation["player_0"]))
        actions[agent_1.player]: np.ndarray = np.array(agent_0.act(step, lux_observation["player_0"]))
        actions_1 = agent_1.act(step, lux_observation["player_1"])
        for unit_index, action in actions_1.items():
            actions[agent_1.player][unit_index] = np.array(action)

        lux_observation, _, terminated, truncated, info = env.step(actions)

        if lux_observation is not None:
            unit_rewards = []
            for unit_index, action in actions_1.items():
                unit = agent_1.observation.player.units[unit_index]
                a = unit.position[0]
                b = unit.position[1]
                state = []
                # center, up, right, down, left
                for x, y in [(a, b), (a, b - 1), (a + 1, b), (a, b + 1), (a - 1, b)]:
                    x = int(x)
                    y = int(y)
                    if x < 0 or y < 0 or x >= agent_1.observation.map.x_max or y >= agent_1.observation.map.y_max \
                            or ((x, y) in agent_1.observation.map.tiles.keys() and agent_1.observation.map.tiles[
                        x, y].type == TILETYPE.ASTEROID):
                        #state.append(torch.inf)
                        state.append(9223372036854775807)
                        continue
                    state.append(agent_1.observation.map.tiles[x, y].index)

                next_observation = Observation(agent_1.team_id, agent_1.opponent_team_id, lux_observation["player_1"], actions=np.array([]))

                unit = next_observation.player.units[unit_index]
                a = unit.position[0]
                b = unit.position[1]
                next_state = []
                # center, up, right, down, left
                for x, y in [(a, b), (a, b - 1), (a + 1, b), (a, b + 1), (a - 1, b)]:
                    x = int(x)
                    y = int(y)
                    if x < 0 or y < 0 or x >= next_observation.map.x_max or y >= next_observation.map.y_max \
                            or ((x, y) in next_observation.map.tiles.keys() and next_observation.map.tiles[
                        x, y].type == TILETYPE.ASTEROID):
                        #next_state.append(torch.inf)
                        next_state.append(9223372036854775807)
                        continue
                    next_state.append(next_observation.map.tiles[x, y].index)

                reward = manhatten_distance((a,b),(0,0)) * -1

                # unit tried to run into asteriod
                act = action[0]
                if (act == DIRECTION.UP and (a,b-1) in agent_1.observation.map.tiles.keys() and agent_1.observation.map.tiles[a,b-1].type == TILETYPE.ASTEROID) or\
                    (act == DIRECTION.RIGHT and (a+1,b) in agent_1.observation.map.tiles.keys() and agent_1.observation.map.tiles[a+1,b].type == TILETYPE.ASTEROID) or\
                    (act == DIRECTION.DOWN and (a,b+1) in agent_1.observation.map.tiles.keys() and agent_1.observation.map.tiles[a,b+1].type == TILETYPE.ASTEROID) or\
                    (act == DIRECTION.LEFT and (a-1,b) in agent_1.observation.map.tiles.keys() and agent_1.observation.map.tiles[a-1,b].type == TILETYPE.ASTEROID):
                    #reward = -1 * torch.inf
                    reward = -1 * 9223372036854775807

                unit_rewards.append(reward)

                agent_1.buffer.add([torch.tensor(state, dtype=torch.double, device=device),
                                    torch.tensor(action[0], dtype=torch.int64, device=device),
                                    torch.tensor(next_state, dtype=torch.double, device=device),
                                    torch.tensor(reward, dtype=torch.double, device=device)])

            if unit_rewards:
                reward_history.append(np.average(unit_rewards))

        agent_1.optimize_model()

        agent_1.soft_update()

        if reward_history:
            # Calculate the Simple Moving Average (SMA) with a window size of 50
            sma = np.convolve(reward_history, np.ones(50) / 50, mode='valid')

            # Clip max (high) values for better plot analysis
            _reward_history = np.clip(reward_history, a_min=None, a_max=100)
            sma = np.clip(sma, a_min=None, a_max=100)

            save_chart("Reward", "Roll Reward", "Step", _reward_history.tolist(), sma.tolist(), [i for i in range(_reward_history.__len__())])

        terminated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})
        truncated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})

        if terminated["player_0"].item() or terminated["player_1"].item() or truncated["player_0"].item() or truncated["player_1"].item():
            done = True
            truncation = True

    env.save_episode(f"test-{episode}.json")