# Research note: large plants and crowds (evacuation step 2)

Steven, 2026-11-21. Written before building; **section 3 items 1-3 are now built**
(`evac.crowd`, `evac.fields`, `evac.crowd_plan`, `uv run evac-crowd`). Results are in
`docs/journey/steven.md`: capacity-aware routing is the best plan with good hazard maps,
and needs a robustness weight with today's baseline maps.

## 1. Why the current planner is not enough

The planner today routes **one person at a time**. Its cost is length plus risk, as if
nobody else were in the hall. Two things break that:

- **Large plants.** On our 60 x 40 m hall nobody walks more than 24 s. On a random
  200 x 100 m plant (`Factory.random(..., width=200, height=100)`) the walk to the nearest
  exit is 39 s at the median, 69 s at the 95th percentile and 81 s at most, still with only
  16 sensors and 4 exits. A real site of several hundred metres means minutes. Over minutes
  the plume moves, the forecast matters far ahead, and replanning starts to pay off (the
  timing study saw no gain from replanning only because walks were short).
- **Crowds.** An exit door lets through a limited flow of people. If everyone picks the
  "best" exit, it jams and people queue in the gas. The fastest route for one person is
  not the best plan for everyone. The plan has to share the doors and corridors.

## 2. What the literature offers

| Family | Idea | Fit for us |
|---|---|---|
| **Dynamic network flows** (Ford & Fulkerson 1958; quickest transshipment, Hoppe & Tardos 2000) | Model the building as a graph with a capacity per edge per second; a time-expanded copy of the graph gives the provably fastest evacuation. | The optimal reference, but the time-expanded graph is large (cells x seconds). Good for small checks of our heuristic. |
| **Capacity-constrained route planning, CCRP** (Lu, George & Shekhar 2005) | Route groups one after another; each takes the earliest-arrival path given the capacity already reserved by the groups before it. Near-optimal, much faster than flows. | **Best fit.** It is our Dijkstra with a reservation table, and adding hazard cost is direct ("risk-aware CCRP"). |
| **Multi-agent path finding** (cooperative A*, Silver 2005; conflict-based search, Sharon et al. 2015) | Collision-free paths for many agents. | Right for the **guide robots** (a few agents, exact positions), too fine-grained for crowds. |
| **Crowd simulation**: floor-field cellular automata (Burstedde et al. 2001), social force (Helbing, Farkas & Vicsek 2000) | Simulate how people actually move, including queues and clogging at doors. | Use it to **score** plans, not to plan: a simple cell-based queue model is enough on our 2 m grid. |

**Capacity numbers to start from:** doors and corridors pass at most about 1.3 people per
metre of clear width per second (pedestrian flow handbooks, e.g. the SFPE Handbook). A 2 m
cell or door therefore passes about 2.6 people/s. Walking speed: 1.3 m/s free, slower in
dense crowds.

## 3. Proposed next build (after the teammates' PRs are merged)

1. **Crowd model to score plans** (`evac/crowd.py`): people as counts per cell. Each cell
   and exit passes at most `capacity` people per second, and the rest queue. Exposure is
   counted per person while queueing. People are sampled where they work (near machines),
   not uniformly.
2. **Risk-aware CCRP planner** (`evac/ccrp.py`): groups are routed in order of urgency
   (most exposed first). The cost is arrival time (including the waiting implied by
   reserved capacity) plus risk weight times P(harm) at arrival. It replans every update
   from where people are.
3. **Compare on the same leaks**, small and large plants, 50-500 people:
   - nearest exit;
   - independent risk Dijkstra (today);
   - risk-aware CCRP.

   Metrics: time until 95 % are out, harm-seconds, people exposed, longest queue per exit.
4. **Repeat the timing study on large plants.** We expect replanning and longer forecast
   horizons to start to matter there. That decides the replan rate and the hazard horizon
   Ahmad must cover.
5. **Robots:** the plan's groups become the guide robots' tasks (which group, which
   route). Multi-agent path finding keeps the robots from blocking each other; the count of
   robots needed comes from the number of groups that must be redirected away from the
   nearest exit.

## References

- L. R. Ford, D. R. Fulkerson. Constructing maximal dynamic flows from static flows.
  *Operations Research* 6(3), 1958.
- B. Hoppe, É. Tardos. The quickest transshipment problem. *Mathematics of Operations
  Research* 25(1), 2000.
- Q. Lu, B. George, S. Shekhar. Capacity constrained routing algorithms for evacuation
  planning: a summary of results. *SSTD 2005*, LNCS 3633.
- D. Silver. Cooperative pathfinding. *AIIDE 2005*.
- G. Sharon, R. Stern, A. Felner, N. Sturtevant. Conflict-based search for optimal
  multi-agent pathfinding. *Artificial Intelligence* 219, 2015.
- C. Burstedde, K. Klauck, A. Schadschneider, J. Zittartz. Simulation of pedestrian dynamics
  using a two-dimensional cellular automaton. *Physica A* 295, 2001.
- D. Helbing, I. Farkas, T. Vicsek. Simulating dynamical features of escape panic.
  *Nature* 407, 2000.
- SFPE Handbook of Fire Protection Engineering, chapter on the movement of people
  (specific flow through doors and corridors).
