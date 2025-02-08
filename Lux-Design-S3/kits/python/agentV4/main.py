import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from typing import TypedDict, Any
from luxai_s3.state import EnvObs, EnvState
import jax
import torch
# custom
from agents import DoNothingAgent
from agents import Agent
from environment import TILETYPE, Observation
from metrics.distance import manhatten_distance

env = LuxAIS3GymEnv()
env = RecordEpisode(env)

lux_observation, info = env.reset()
lux_observation: TypedDict('Observation', {'player_0': EnvObs, "player_1": EnvObs}) = lux_observation
info: TypedDict('Info', {'params': dict[str, int], "full_params": dict[str, Any], "state": EnvState}) = info
done = False
truncation = False
agent_0 = DoNothingAgent("player_0", info["params"])
agent_1 = Agent("player_1", info["params"])

step = 0
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
                    state.append(torch.inf)
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
                    next_state.append(torch.inf)
                    continue
                next_state.append(next_observation.map.tiles[x, y].index)

            reward = manhatten_distance((a,b),(0,0))

            agent_1.buffer.add([state, action, next_state, reward])

    agent_1.optimize_model()

    agent_1.soft_update()

    terminated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})
    truncated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})

    if terminated["player_0"].item() or terminated["player_1"].item() or truncated["player_0"].item() or truncated["player_1"].item():
        done = True
        truncation = True

env.save_episode("test.json")