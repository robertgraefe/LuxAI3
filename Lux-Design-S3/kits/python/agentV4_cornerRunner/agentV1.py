import numpy as np
from agentV1_utils.environment import EnvironmentConfig, Observation, TILETYPE
from loguru import logger
import random
import  math
logger.remove()
logger.add("game.log", enqueue=False, mode="w", format="{message}")

# https://www.kaggle.com/code/zakirkhanaleemi/lux-ai-season-starter-notebook/notebook
# cd C:\Users\rober\Documents\Workspace\LuxAI3\Lux-Design-S3\kits\python
# luxai-s3 main.py main.py --output replay.json

# cd /home/robert/PycharmProjects/LuxAI3/Lux-Design-S3/kits/python
# conda activate LuxAI
# luxai-s3 agentV1/main.py agentV2/main.py --output replay.json

class Agent:
    last_observation: Observation = None

    def __init__(self, player: str, env_cfg) -> None:
        logger.info("#### GAME STARTS ####")
        self.env_cfg = EnvironmentConfig(**env_cfg)
        self.player = player
        self.opponent = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0

    def act(self, step: int, obs, remainingOverageTime: int = 60):

        actions = np.zeros((self.env_cfg.max_units, 3), dtype=int)

        if self.team_id == 1:
            return actions

        observation = Observation(self.team_id, self.opponent_team_id, obs, actions, self.last_observation)


        if [tile.position for tile in observation.map.tiles.values() if tile.is_relic]:
            logger.info([tile.position for tile in observation.map.tiles.values() if tile.is_relic])


        for unit in observation.player.units.values():
            nearest_known_point = unit.position
            for point in observation.map.tiles.keys():
                if observation.map.tiles[point].type != TILETYPE.SPACE:
                    continue
                if unit.destination is None or point is None or nearest_known_point is None:
                    continue
                if observation.map.manhatten_distance(unit.destination, point) < observation.map.manhatten_distance(unit.destination,
                                                                                                      nearest_known_point):
                    nearest_known_point = point

            path = observation.map.shortest_path(unit.position, nearest_known_point)
            if path:
                unit.move_position(path[1])

        if [tile.position for tile in observation.map.tiles.values() if tile.is_relic]:
            for i, unit in enumerate(observation.player.units.values()):
                if i % 2 == 0:
                    relic = random.choice([tile for tile in observation.map.tiles.values() if tile.is_relic])
                    unit.destination = relic.position




        self.last_observation = observation

        return actions
