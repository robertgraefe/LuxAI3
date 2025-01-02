from typing import Any

from luxai_s3.wrappers import LuxAIS3GymEnv
from tf_agents.environments import PyEnvironment
from tf_agents.trajectories import TimeStep
from tf_agents.typing import types


class LuxAIS3PyEnv(PyEnvironment):
    gymEnv = LuxAIS3GymEnv()

    def action_spec(self) -> types.NestedArraySpec:
        return self.gymEnv.action_space

    def get_info(self) -> types.NestedArray:
        return self.gymEnv.metadata

    def get_state(self) -> Any:
        return self.gymEnv.state

    def set_state(self, state: Any) -> None:
        self.gymEnv.state = state

    def _step(self, action: types.NestedArray) -> TimeStep:
        obs, reward, terminated, truncated, info = self.gymEnv.step(action)
        # step_type, reward, discount, observation
        ts = TimeStep(None, reward, None, obs)
        return ts

    def _reset(self) -> TimeStep:
        obs, paramDict = self.gymEnv.reset()
        return TimeStep(None, None, None, obs)

    def observation_spec(self) -> types.NestedArraySpec:
        return self.gymEnv.observation_space