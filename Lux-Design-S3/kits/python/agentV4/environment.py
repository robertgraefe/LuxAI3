import numpy as np
from dataclasses import dataclass
import random

from luxai_s3.state import EnvObs
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from actions import DIRECTION

"""
{
  "max_units": 16,
  "match_count_per_episode": 5,
  "max_steps_in_match": 100,
  "map_height": 24,
  "map_width": 24,
  "num_teams": 2,
  "unit_move_cost": 5,
  "unit_sap_cost": 46,
  "unit_sap_range": 4,
  "unit_sensor_range": 4
}
"""


class EnvironmentConfig:
    def __init__(self, **kwargs):
        self.max_units: int = kwargs.pop("max_units")
        self.match_count_per_episode: int = kwargs.pop("match_count_per_episode")
        self.max_steps_in_match: int = kwargs.pop("max_steps_in_match")
        self.map_height: int = kwargs.pop("map_height")
        self.map_width: int = kwargs.pop("map_width")
        self.num_teams: int = kwargs.pop("num_teams")
        self.unit_move_cost: int = kwargs.pop("unit_move_cost")
        self.unit_sap_cost: int = kwargs.pop("unit_sap_cost")
        self.unit_sap_range: int = kwargs.pop("unit_sap_range")
        self.unit_sensor_range: int = kwargs.pop("unit_sensor_range")


class Unit:
    id: int
    position: np.ndarray
    x: int
    y: int
    energy: int
    actions: np.array

    def __init__(self):
        self.destination: np.ndarray or None = None

    def move_direction(self, direction: DIRECTION):
        self.actions[self.id] = [direction, 0, 0]

    def move_position(self, position: np.ndarray):
        if position is None:
            return
        ds = position - self.position
        dx = ds[0]
        dy = ds[1]
        if abs(dx) > abs(dy):
            if dx > 0:
                direction = DIRECTION.RIGHT
            else:
                direction = DIRECTION.LEFT
        else:
            if dy > 0:
                direction = DIRECTION.DOWN
            else:
                direction = DIRECTION.UP
        self.actions[self.id] = [direction, 0, 0]


class Player:
    points: int
    wins: int
    units: dict[int, Unit] = {}


@dataclass
class TILETYPE:
    UNKOWN = -1
    SPACE = 0
    NEBOLA = 1
    ASTEROID = 2


class Tile:
    position: np.ndarray
    x: int
    y: int
    type: TILETYPE
    index: int
    is_relic: bool = False


class Map:
    energy: np.ndarray
    tile_types: np.ndarray[np.ndarray]
    tiles = dict[tuple, Tile]
    last_map: 'Map'
    x_max: int
    y_max: int
    adj_matrix: np.ndarray

    def __init__(self):
        self.tiles: dict[tuple, Tile] = {}

    @property
    def unkown_tiles(self) -> list[Tile]:
        return [tile for tile in self.tiles.values() if tile.type == TILETYPE.UNKOWN]

    @property
    def asteriod_tiles(self) -> list[Tile]:
        return [tile for tile in self.tiles.values() if tile.type == TILETYPE.ASTEROID]

    @property
    def display_tile_types(self) -> np.ndarray[np.ndarray]:
        return self.tile_types.copy().transpose()

    @property
    def display_anticipated_tile_types(self) -> np.ndarray:
        result = self.tile_types.copy()
        x_max, y_max = self.tile_types.shape
        for x in range(x_max):
            for y in range(y_max):
                result[x, y] = self.tiles[x, y].type
        return result.transpose()

    def get_adj_matrix(self):
        n = len(self.tiles)
        matrix = np.zeros((n, n))
        for index, tile in enumerate(self.tiles.values()):
            if tile.type != TILETYPE.SPACE:
                continue
            adj = [(tile.x + 1, tile.y), (tile.x - 1, tile.y), (tile.x, tile.y + 1), (tile.x, tile.y - 1)]
            for point in adj:
                if point in self.tiles:
                    if self.tiles[point].type == TILETYPE.SPACE:
                        matrix[index, self.tiles[point].index] = 1
        dist_matrix, pred = shortest_path(csr_matrix(matrix), return_predecessors=True)
        self.adj_matrix = pred

    def __get_path(self, pred_matrix: np.ndarray, start: int, end: int) -> list:
        path = []
        current = end
        while pred_matrix[start, current] != -9999:
            path.append(pred_matrix[start, current])
            current = pred_matrix[start, current]
        if path:
            path = path[::-1]
            path.append(np.int32(end))
        return path


    def shortest_path(self, start: tuple[int, int], end: tuple[int, int]):
        sx, sy = start
        x_max, y_max = self.tile_types.shape
        start_index = sx * x_max + sy
        ex, ey = end
        end_index = ex * x_max + ey
        path = self.__get_path(self.adj_matrix, start_index, end_index)
        return [(np.floor(index / x_max), index % x_max) for index in path]

    def __index_to_point(self, index: int) -> tuple[int, int]:
        return np.floor(index / self.x_max).item(), index % self.x_max

    def manhatten_distance(self, point_a: tuple[int,int], point_b: tuple[int,int]):
        ax,ay = point_a
        bx,by = point_b
        return abs(ax-bx) + abs(ay-by)

    @property
    def map_movement_direction(self) -> DIRECTION:
        if self.last_map is None:
            return DIRECTION.CENTER

        mask1 = np.where(self.last_map.tile_types != TILETYPE.UNKOWN)
        mask2 = np.where(self.tile_types != TILETYPE.UNKOWN)
        m1 = [(x, y) for x, y in zip(mask1[0], mask1[1])]
        m2 = [(x, y) for x, y in zip(mask2[0], mask2[1])]
        diff = [point for point in m1 if point in m2]
        rows = [point[0] for point in diff]
        cols = [point[1] for point in diff]

        current = self.tile_types[rows, cols]
        last = self.last_map.tile_types[rows, cols]

        if np.all(current == last):
            return DIRECTION.CENTER

        if self.__get_movement(-1, 1):
            return DIRECTION.DOWN_LEFT
        if self.__get_movement(1, -1):
            return DIRECTION.UP_RIGHT
        if self.__get_movement(0, 1):
            return DIRECTION.DOWN
        if self.__get_movement(1, 1):
            return DIRECTION.DOWN_RIGHT
        if self.__get_movement(-1, 0):
            return DIRECTION.LEFT
        if self.__get_movement(1, 0):
            return DIRECTION.RIGHT
        if self.__get_movement(-1, -1):
            return DIRECTION.UP_LEFT
        if self.__get_movement(0, -1):
            return DIRECTION.UP
        return DIRECTION.CENTER

    def __get_movement(self, a: int, b: int) -> bool:
        if self.last_map is None:
            return False

        a = a * -1
        b = b * -1

        mask1 = np.where(self.last_map.tile_types != TILETYPE.UNKOWN)
        mask2 = np.where(self.tile_types != TILETYPE.UNKOWN)
        m1 = [(x, y) for x, y in zip(mask1[0], mask1[1])]
        m2 = [(x, y) for x, y in zip(mask2[0], mask2[1])]
        diff = [point for point in m1 if point in m2]

        mask = [(x, y) for x, y in diff if (x + a, y + b) in diff]
        mask_rows = [x for x, y in mask]
        mask_cols = [y for x, y in mask]

        last_copy = self.last_map.tile_types.copy()
        for x, y in mask:
            last_copy[x, y] = self.last_map.tile_types[x + a, y + b]

        return np.all(self.tile_types[mask_rows, mask_cols] == last_copy[mask_rows, mask_cols])


class Observation:
    def __init__(self, player_team_id: int, opponent_team_id: int, obs: EnvObs, actions: np.ndarray,
                 last_observation: 'Observation' = None):
        # Players
        self.player = Player()
        self.opponent = Player()

        self.player.points = obs.team_points[player_team_id].item()
        self.opponent.points = obs.team_points[opponent_team_id].item()
        self.player.wins = obs.team_wins[player_team_id].item()
        self.opponent.wins = obs.team_wins[opponent_team_id].item()

        # SENSOR MASK
        #self.sensor_mask = np.array(obs["sensor_mask"])
        self.sensor_mask = np.array(obs.sensor_mask)

        # MAP
        map_features = Map()
        map_features.energy = np.array(obs.map_features.energy)
        map_features.tile_types = np.array(obs.map_features.tile_type)
        map_features.last_map = last_observation.map if last_observation else None
        map_features.tiles = last_observation.map.tiles if last_observation else {}
        map_features.x_max, map_features.y_max = map_features.tile_types.shape
        map_features.get_adj_matrix()
        self.map = map_features

        # TILES

        x_max, y_max = self.map.tile_types.shape
        for x in range(x_max):
            for y in range(y_max):
                type = self.map.tile_types[x, y]

                if type == TILETYPE.UNKOWN and last_observation:
                    type = last_observation.map.tiles[x, y].type


                tile = Tile()
                tile.x = x
                tile.y = y
                tile.position = np.array([x, y])
                tile.type = type
                tile.index = y + x * x_max
                self.map.tiles[x, y] = tile

        # RELIC NODES
        self.relic_nodes = np.array(obs.relic_nodes)
        self.relic_nodes_mask = np.array(obs.relic_nodes_mask)

        rows,cols = np.where(self.relic_nodes > [-1, -1])
        if rows.any() and cols.any():
            relic_points = self.relic_nodes[np.unique(rows)]
            for point in relic_points:
                self.map.tiles[point[0].item(), point[1].item()].is_relic = True

        if last_observation:
            rows, cols = np.where(last_observation.relic_nodes > [-1, -1])
            if rows.any() and cols.any():
                relic_points = last_observation.relic_nodes[np.unique(rows)]
                for point in relic_points:
                    self.map.tiles[point[0].item(), point[1].item()].is_relic = True

        self.steps: int = obs.steps
        self.match_steps: int = obs.match_steps

        # UNITS - Player
        unit_ids = np.where(obs.units_mask[player_team_id])[0]
        for unit_id in unit_ids:
            unit = Unit()
            unit.id = unit_id
            unit.position = np.array(obs.units.position[player_team_id])[unit_id]
            unit.x = unit.position[0]
            unit.y = unit.position[1]
            unit.energy = np.array(obs.units.energy[player_team_id])[unit_id]
            unit.actions = actions

            if (self.map.unkown_tiles):
                destination = random.choice(self.map.unkown_tiles).position
                if last_observation and unit_id in last_observation.player.units:
                    destination = last_observation.player.units[unit_id].destination
                unit.destination = destination
            else:
                unit.destination = random.choice(self.map.asteriod_tiles).position


            self.player.units[unit_id] = unit

        # UNITS - Opponent
        unit_ids = np.where(obs.units_mask[opponent_team_id])[0]
        for unit_id in unit_ids:
            unit = Unit()
            unit.id = unit_id
            unit.position = np.array(obs.units.position[opponent_team_id])[unit_id]
            unit.x = unit.position[0]
            unit.y = unit.position[1]
            unit.energy = np.array(obs.units.energy[opponent_team_id])[unit_id]
            unit.actions = actions
            self.opponent.units[unit_id] = unit
