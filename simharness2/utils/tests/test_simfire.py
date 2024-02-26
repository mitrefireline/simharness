import unittest
from simfire.utils.config import Config
from simharness2.utils.simfire import parse_config


class TestSimFire(unittest.TestCase):
    def setUp(self):
        # self.cfg = Config(
        #     {
        #         "area": {"width": 1000, "height": 500},
        #         "simulation": {"duration": 3600, "time_step": 60},
        #         "mitigation": {"firebreak_width": 10, "firebreak_cost": 1000},
        #         "operational": {"num_personnel": 10, "num_vehicles": 2},
        #         "terrain": {"type": "flat", "elevation": 0},
        #         "environment": {"temperature": 25, "humidity": 50},
        #         "wind": {"speed": 10, "direction": 0},
        #     }
        # )
        pass

    def test_parse_config(self):
        # expected = {
        #     "area": {"width": 1000, "height": 500},
        #     "simulation": {"duration": 3600, "time_step": 60},
        #     "mitigation": {"firebreak_width": 10, "firebreak_cost": 1000},
        #     "operational": {"num_personnel": 10, "num_vehicles": 2},
        #     "terrain": {"type": "flat", "elevation": 0},
        #     "topography": {},
        #     "fuel": {},
        #     "fire": {},
        #     "environment": {"temperature": 25, "humidity": 50},
        #     "wind": {"speed": 10, "direction": 0},
        # }
        # self.assertEqual(parse_config(self.cfg), expected)
        pass

    def test_get_area_params(self):
        # flat_cfg = {"area.width": 1000, "area.height": 500}
        # expected = {"width": 1000, "height": 500}
        # self.assertEqual(_get_area_params(flat_cfg), expected)
        pass

    # Add more tests for the other functions in simfire.py


if __name__ == "__main__":
    unittest.main()
