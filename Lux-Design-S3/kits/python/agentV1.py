import numpy as np
import logging
from environment import EnvironmentConfig

logger = logging.getLogger(__name__)
logging.basicConfig(filename='game.log', level=logging.INFO, filemode="w")


class Agent:

    def __init__(self, player: str, env_cfg) -> None:
        logger.info("#### GAME STARTS ####")
        self.env_cfg = EnvironmentConfig(**env_cfg)
        self.player = player
        self.opponent = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        return np.zeros((self.env_cfg.max_units, 3), dtype=int)
