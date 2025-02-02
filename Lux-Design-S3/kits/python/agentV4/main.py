from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from typing import TypedDict, Any
from luxai_s3.state import EnvObs, EnvState
import jax
# custom
from agents import DoNothingAgent
from agents import Agent

env = LuxAIS3GymEnv()
env = RecordEpisode(env)

observation, info = env.reset()
observation: TypedDict('Observation', {'player_0': EnvObs, "player_1": EnvObs}) = observation
info: TypedDict('Info', {'params': dict[str, int], "full_params": dict[str, Any], "state": EnvState}) = info
done = False
truncation = False
agent_0 = DoNothingAgent("player_0", info["params"])
agent_1 = Agent("player_1", info["params"])

step = 0
while not done and not truncation:
    step += 1
    actions = dict()

    actions[agent_0.player] = agent_0.act(step, observation["player_0"])
    actions[agent_1.player] = agent_1.act(step, observation["player_1"])

    observation, _, terminated, truncated, info = env.step(actions)

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
                agent_1.buffer.push(state, action, next_state, reward)

            optimize_model()

            agent_1.soft_update()

            terminated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})
            truncated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})

            if terminated["player_0"].item() or terminated["player_1"].item() or truncated["player_0"].item() or truncated["player_1"].item():
                done = True
                truncation = True

env.save_episode("test.json")