import pytest
from simharness2.agents import ReactiveAgent


class TestReactiveAgent:
    def test_post_init(self):
        agent = ReactiveAgent(agent_id="agent_5", sim_id=5, initial_position=(0, 20))
        # Validate initialization of attrs
        assert agent.initial_position == (0, 20)
        assert agent.latest_movement is None
        assert agent.latest_interaction is None
        assert not agent.mitigation_placed
        assert not agent.moved_off_map

        # Validate initialization of properties
        assert agent.current_position == agent.initial_position
        assert agent.x == 0
        assert agent.y == 20
        assert agent.row == 20
        assert agent.col == 0

    def test_current_position(self):
        agent = ReactiveAgent(agent_id="agent_5", sim_id=5, initial_position=(0, 20))
        agent.current_position = (30, 40)
        assert agent.current_position != agent.initial_position
        assert agent.x == 30
        assert agent.y == 40
        assert agent.row == 40
        assert agent.col == 30

    # def test_xy_properties(self):
    #     agent = ReactiveAgent(agent_id="agent_5", sim_id=5, initial_position=(0, 20))
    #     agent.x = 30
    #     assert agent.current_position == (30, 20)
    #     agent.y = 40
    #     assert agent.current_position == (30, 40)


if __name__ == "__main__":
    pytest.main()
