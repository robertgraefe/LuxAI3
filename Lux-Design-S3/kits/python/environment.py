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
