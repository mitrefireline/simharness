DEFAULT_SIMFIRE_CONFIG_CLASS = "simfire.utils.config.Config"
DEFAULT_SIMFIRE_SIMULATION_CLASS = "simfire.sim.simulation.FireSimulation"

NOOP_KEY = "none"
# movements
MOVEMENTS = ["up", "down", "left", "right"]
MOVEMENTS_WITH_NONE = [NOOP_KEY] + MOVEMENTS

# interactions
INTERACTIONS = ["fireline"]
INTERACTIONS_WITH_NONE = [NOOP_KEY] + INTERACTIONS
FULL_INTERACTIONS = ["fireline", "scratchline", "wetline"]
FULL_INTERACTIONS_WITH_NONE = [NOOP_KEY] + FULL_INTERACTIONS

# analytics
DEFAULT_HARNESS_ANALYTICS_CLASS = (
    "simharness2.analytics.harness_analytics.ReactiveHarnessAnalytics"
)
DEFAULT_SIM_ANALYTICS_CLASS = (
    "simharness2.analytics.simulation_analytics.FireSimulationAnalytics"
)
DEFAULT_AGENT_ANALYTICS_CLASS = (
    "simharness2.analytics.agent_analytics.ReactiveAgentAnalytics"
)
