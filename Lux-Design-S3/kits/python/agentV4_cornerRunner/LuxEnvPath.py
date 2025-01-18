from typing import Any, SupportsFloat
import jax
from luxai_s3.wrappers import LuxAIS3GymEnv


class LuxEnvPath(LuxAIS3GymEnv):

    def __init__(self):
        super().__init__()


    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:

        observation, reward, terminated, truncated, info = super().step(action)

        reward = {
            "player_0": jax.numpy.array(1),
            "player_1": jax.numpy.array(1)
        }

        return observation, reward, terminated, truncated, info

