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