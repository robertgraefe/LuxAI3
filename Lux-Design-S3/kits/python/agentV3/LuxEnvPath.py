from typing import Any, SupportsFloat, TypedDict
import jax
import numpy as np
from luxai_s3.state import EnvObs
from luxai_s3.wrappers import LuxAIS3GymEnv
from agentV1_utils.environment import Observation

class LuxEnvPath(LuxAIS3GymEnv):

    def __init__(self):
        super().__init__()


    def step(
        self, actions: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        observation, reward, terminated, truncated, info = super().step(actions)

        observation: TypedDict('Observation', {'player_0': EnvObs, "player_1": EnvObs}) = observation

        observation_player_0 = Observation(0, 1, observation["player_0"], actions)
        observation_player_1 = Observation(1, 0, observation["player_1"], actions)

        destination = (0,23)



        reward = {
            "player_0": jax.numpy.array(1),
            "player_1": jax.numpy.array(1)
        }

        return observation, reward, terminated, truncated, info

