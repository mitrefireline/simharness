# Realistic Agent Dynamics

## General Notes

- In building fireline, all fuels are removed and the surface is scraped to mineral soil on a **strip between 6 inches and 3 feet wide**, depending upon the fuel and slope.
  - To make things easier, we can just assume that agents are digging firelines that are w/in the 6 inches to 3 feet bound. This won't affect calculations for MVP v1.
  - [Source](https://www.nps.gov/articles/wildland-fire-fireline-construction.htm)
- `self.sim.elapsed_time` is in **minutes**
- Each pixel is a 30m x 30m = 900m of area to dig
- For meters to feet, do: `X meters / 0.3048 = X feet`
- Thus, each pixel is approximately `2953 ft` of land (rounding up)
- For the above, we actually have a method already for meters to feet:

```python
from simfire.utils.units import meters_to_feet

# Note that each pixel is 30m x 30m (based on Landfire Data Layers).
pixel_area_in_feet = meters_to_feet(30 * 30)

# Output: pixel_area_in_feet == 2952.756
```

- What data should be used from the "Fireline Production Rate Tables"?
  - In [this](https://www.frames.gov/documents/behaveplus/publications/NWCG_2021_FireLineProductionRates.pdf) pdf,
    there are 9 or so data tables to look at. So, we have to decide where to start.
  - For "Sustained Line Production Rates", this seems to map towards the agent selecting to interact with a FIRELINE
    mitigation.
  - For "Line Production Rates for Initial Action", this seems to map towards the agent selecting to interact with
    a SCRATCHLINE mitigation. The reason is that there is a note under the data table that states:

> Do not use these rates to estimate sustained line construction, burnout, and holding productivity. Initial action
> may consist of scratch line construction and hotspotting.

For the MVP v1, we will make a few assumptions:

- Agents represent a hand crew. We need to determine the amount of people each person represents, ie. probably doesn't make sense for 1 agent to equal 1 person, but also doesn't make sense for 1 agent to equal 20 people.
- We will not support dozers, tractor, plows, etc. at this time. The only agent type is "hand crew".
- The only mitigation type available to agents is FIRELINE. The interaction space will be decided with preliminary testing, as I'm not sure if we will or will not want to incorporate the NOOP action.
  - In MVP v2, it would be interesting to introduce SCRATCHLINE as a mitigation type. These are the "initial action" mitigations performed, and they should take the agent less time to complete. So, in theory, maybe the advantage of saving time would allow the agent to learn when to use each mitigation type. But as stated, this will be left for a future implementation.
  - Note that in simfire, if `mitigation.ros_attenuation` is set to `False`, a SCRATCHLINE and WETLINE will have the same dampening factor for the rate of spread!
- For data, we will solely focus on the "Sustained Line Production Rates" table (s).
- Each pixel in the simfire `fire_map` has a fuel value (when using Landfire operational data layers). The agent **CANNOT** place a mitigation on pixels that have a `Fuel` name contained in `[NBUrban, NBSnowIce, NBAgriculture, NBWater, NBBarren, NBNoData]`.
  - NB == Nonburnable.
  - Ideally, we would perform action masking here to completely disallow the model from selecting a mitigation action when on a pixel with the respective value. However, this will entail a more complex implementation. Anything involving "action masking" will be delayed to future, ie. potentially in MVP v2.
- The TYPE of fire crew will by TYPE 1, and control lines will always be DIRECT. We can adjust this after MVP v1 is complete.


Interesting note about crew starting locations (source: [Handcrews](https://www.fs.usda.gov/science-technology/fire/people/handcrews)):
> All crews begin constructing fireline from a safe anchor point. An anchor point is a natural or human made area that cannot burn. A road, lake, stream, rock outcropping, or even another secure fireline are good examples. Anchor points reduce the chances of crews being flanked by the fire while the fireline is being constructed.

- I wonder if this is something we should try to leverage, or offer as an option, for selecting agent start positions. When operational fuel is used as a data layer, we can find all possible pixels that fall into the **NB** fuel type with relative ease. Maybe this is used to generate candidate start positions for each agent, and then some sort of sampling happens or additional logic to choose the agent start location?

## Decision Points

- How many "people" does a single agent represent? This can be empirically tested, as the only change factor will be how we adjust the rate tables based on the size of the crew (for a single agent).
- Can the agent choose the NOOP interaction, allowing it to avoid placing a mitigation in favor of moving elsewhere on the map? I suspect this will be crucial, otherwise the agent won't be able to cover ground as the fire propagation progresses.
- Do we want to "re-structure" the action space to allow the agent to only select "move" or "mitigate" at each timestep? With the agent now restricted by time when placing a mitigation, I'm not sure we need the agent to "always" choose both interact and movement.
  - If each "action step" in the simulation is 1 minute (which can be adjusted), then it would make sense to only make one action decision each minute. Note that things change if `agent_speed` is not equal to `1`!
- If an agent chooses to place a mitigation, *when* should we call `sim.update_mitigation()` so that the mitigation is represented as "finished" in simfire? I *think* it makes the most sense to do this once the agent has "finished digging"?


## Definitions

- **wet line**: A line of water, or water and chemical retardant, sprayed along the ground, and which serves as a temporary control line from which to ignite or stop a low-intensity fire.
- **fireline (line)**: The part of a containment or control line that is scraped or dug to mineral soil.
- **scratch line**: An unfinished preliminary control line hastily established or constructed as an emergency measure to check the spread of fire.
- **direct attack**: Any treatment applied directly to burning fuel such as wetting, smothering, or chemically quenching the fire or by physically separating the burning from unburned fuel.
- **indirect attack**: A method of suppression in which the control line is located some considerable distance away from the fire's active edge. Generally done in the case of a fast-spreading or high-intensity fire and to utilize natural or constructed firebreaks or fuel breaks and favorable breaks in the topography. The intervening fuel is usually backfired; but occasionally the main fire is allowed to burn to the line, depending on conditions.

([source](https://www.nwcg.gov/publications/pms205/nwcg-glossary-of-wildland-fire-pms-205))

- **FuelModelToFuel** (see [here](https://github.com/mitrefireline/simfire/blob/5d76a16de45e7058e5080f2cf4c5b5b2f9f1d0ae/simfire/enums.py#L176) in `simfire`) defines the expected fuel values that a single pixel must have when using operational fuel:

```json
FuelModelToFuel = {
    1: ShortGrass,
    2: GrassTimberShrubOverstory,
    3: TallGrass,
    4: Chaparral,
    5: Brush,
    6: DormantBrushHardwoodSlash,
    7: SouthernRough,
    8: ClosedShortNeedleTimberLitter,
    9: HardwoodLongNeedlePineTimber,
    10: TimberLitterUnderstory,
    11: LightLoggingSlash,
    12: MediumLoggingSlash,
    13: HeavyLoggingSlash,
    91: NBUrban,
    92: NBSnowIce,
    93: NBAgriculture,
    98: NBWater,
    99: NBBarren,
    -32768: NBNoData,
    -9999: NBNoData,
    32767: NBNoData,
}
```

- **NB Fuel**: Insufficient wildland fuel to carry wildland fire under any condition (Nonburnable). In all NB fuel models there is no fuel load - **wildland fire will not spread** (therefore, no need to perform a mitigation). For more details on "Nonburnable Fuel Type Models (NB)", see page 19 to 24 [here](https://gacc.nifc.gov/oncc/docs/40-Standard%20Fire%20Behavior%20Fuel%20Models.pdf).

  1. `NBUrban`: Urban or suburban development; insufficient wildland fuel to carry wildland fire.
  2. `NBSnowIce`: Snow/ice.
  3. `NBAgriculture`: Agricultural field, maintained in nonburnable condition.
  4. `NBWater`: Open water.
  5. `NBBarren`: Bare ground.

## Miscellaneous Questions

- What is the simfire `FuelParticle` used for? See [here](https://github.com/mitrefireline/simfire/blob/5d76a16de45e7058e5080f2cf4c5b5b2f9f1d0ae/simfire/world/parameters.py#L8) in simfire repository.
