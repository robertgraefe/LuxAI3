import jax
from luxai_s3.state import EnvObs, EnvState
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from typing import TypedDict, Any
from agentV1 import Agent
from LuxEnvPath import LuxEnvPath

env = LuxAIS3GymEnv()

# env = LuxEnvPath()
env = RecordEpisode(env)


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
while not done and not truncation:
    actions = dict()
    actions[agent_0.player] = agent_0.act(step, observation["player_0"])
    actions[agent_1.player] = agent_1.act(step, observation["player_1"])

    observation, reward, terminated, truncated, info = env.step(actions)
    terminated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})
    truncated: TypedDict("Terminated", {"player_0": jax.Array, "player_1": jax.Array})

    if terminated["player_0"].item() or terminated["player_1"].item() or truncated["player_0"].item() or truncated["player_1"].item():
        done = True
        truncation = True

    step += 1

env.save_episode("test.json")
#env.close()