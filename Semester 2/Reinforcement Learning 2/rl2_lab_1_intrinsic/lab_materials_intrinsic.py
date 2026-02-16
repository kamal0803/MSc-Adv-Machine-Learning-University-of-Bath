import textwrap
import random
from typing import Tuple, Dict, List, Callable, Protocol, runtime_checkable, Optional

import numpy as np
import matplotlib.pyplot as plt

State = Tuple[int, int]  # (y, x)
Action = int


class GridWorldEnv(object):
    def __init__(
        self,
        layout="""
        ###############################
        #.............#...#....#......#
        #.............#...#....#...#..#
        #.............#........#...#..#
        #.............#...#....#......#
        #.............#...#....#...#..#
        #.............#........#...#..#
        #.............#...#........#..#
        #.............#####....##.##..#
        #.............#...#....#...#..#
        #.............#...#....#...#..#
        #.............#...#....#......#
        #.............#...#....#...#..#
        #.............#...#....#...#..#
        #.................#....#......#
        #.............#...######...#..#
        ################.##....##.##..#
        #.............#...#....#...#..#
        #.................#....#...#..#
        #.............#...#....#...#..#
        ###.#########.#####....#......#
        #....#..........#......#...#..#
        #....#..........####.###...#..#
        #....#..........#......##.##..#
        #......................#...#..#
        #....#..........#.............#
        #....#..........#......#...#..#
        ###############################
        """,
        start_pos=(1, 1),
        goal_pos=(4, 16),
    ):
        # Parse the layout into a 2D numpy array.
        dedented = textwrap.dedent(layout)
        self.layout = np.array([list(line) for line in dedented.strip().splitlines()])
        self.height, self.width = self.layout.shape

        # Store start and goal positions.
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # Initialise agent position.
        self.agent_pos = None

        # Define the number of actions: right, down, left, up.
        self.num_actions = 4

        # Persistent plotting objects.
        self._fig = None
        self._ax = None
        self._im = None

    def reset(self) -> State:
        # Reset the agent to the start position.
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[State, float, bool]:
        # Define action effects.
        action_effects = {
            0: (0, 1),  # East
            1: (1, 0),  # South
            2: (0, -1),  # West
            3: (-1, 0),  # North
        }

        # Get the effect of the action.
        if action in action_effects:
            dy, dx = action_effects[action]
            new_pos = (self.agent_pos[0] + dy, self.agent_pos[1] + dx)

            # Check for wall collisions.
            if self.layout[new_pos] != "#":
                self.agent_pos = new_pos

        # Check if the goal is reached.
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01

        return self.agent_pos, reward, done

    def render(self):
        # Create an RGB image and paint cells by precedence:
        # agent > goal > start > wall > floor
        h, w = self.layout.shape
        img = np.ones((h, w, 3), dtype=float)  # default: white floor

        # Walls: black
        walls = self.layout == "#"
        img[walls] = [0.0, 0.0, 0.0]

        # Start: green
        if self.start_pos is not None:
            img[self.start_pos] = [0.0, 1.0, 0.0]

        # Goal: red
        if self.goal_pos is not None:
            img[self.goal_pos] = [1.0, 0.0, 0.0]

        # Agent: blue (overrides start/goal for visibility)
        if self.agent_pos is not None:
            img[self.agent_pos] = [0.0, 0.0, 1.0]

        # If figure not created yet, create and store handles so we reuse one window
        if self._fig is None or self._ax is None or self._im is None:
            self._fig, self._ax = plt.subplots()
            self._im = self._ax.imshow(img, interpolation="nearest")
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._fig.tight_layout()
        else:
            self._im.set_data(img)

        # Notebook vs script display behaviour
        try:
            from IPython.display import clear_output, display
            from IPython import get_ipython

            ip = get_ipython()
            if ip is not None:
                clear_output(wait=True)
                display(self._fig)
                plt.close(self._fig)
                self._fig.canvas.draw()
                return
        except Exception:
            pass

        plt.ion()
        self._fig.canvas.draw_idle()
        plt.pause(0.001)


# ============================================================
# Helper utilities (used by the notebook)
# ============================================================


@runtime_checkable
class RLAgent(Protocol):
    """Minimal protocol for agents used by the helper functions in this file."""

    def select_action(self, s: State) -> Action: ...
    def update(self, s: State, a: Action, r_extrinsic: float, s_next: State, terminal: bool) -> float: ...


def set_global_seeds(seed: int) -> None:
    """Set Python + NumPy RNG seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)


def collect_visitation_counts(env: GridWorldEnv, agent: RLAgent, *, num_steps: int) -> np.ndarray:
    """
    Run the agent for exactly `num_steps` environment steps (resetting on terminal),
    updating the agent online, and return visit counts over states.

    Returns:
        counts: array of shape (env.height, env.width)
    """
    H, W = env.height, env.width
    counts = np.zeros((H, W), dtype=np.int64)

    s = env.reset()
    y, x = s
    counts[y, x] += 1

    for _ in range(num_steps):
        a = agent.select_action(s)
        s_next, r_ext, terminal = env.step(a)

        agent.update(s, a, float(r_ext), s_next, terminal)

        s = s_next
        y, x = s
        counts[y, x] += 1

        if terminal:
            s = env.reset()
            y, x = s
            counts[y, x] += 1

    return counts


def average_visitation_counts(
    make_env: Callable[[], GridWorldEnv],
    make_agent: Callable[[GridWorldEnv], RLAgent],
    *,
    num_steps: int = 500,
    num_runs: int = 5,
    base_seed: int = 0,
) -> np.ndarray:
    """Average visitation counts over multiple random seeds."""
    all_counts: List[np.ndarray] = []

    for i in range(num_runs):
        set_global_seeds(base_seed + i)
        env = make_env()
        agent = make_agent(env)
        all_counts.append(collect_visitation_counts(env, agent, num_steps=num_steps).astype(np.float64))

    return np.mean(all_counts, axis=0)


def plot_visitation_heatmaps(
    counts_by_label: Dict[str, np.ndarray],
    *,
    title: str = "Visitation heatmaps",
    log_scale: bool = True,
    per_panel_normalise: bool = True,
    shared_colour_scale: bool = True,
    zero_outer_border: bool = True,
) -> None:
    """
    Plot visitation heatmaps side-by-side.

    - If `per_panel_normalise=True`, each heatmap is scaled to [0,1] independently.
    - If `per_panel_normalise=False` and `shared_colour_scale=True`, all panels share the same colour scale.
    """
    labels = list(counts_by_label.keys())
    arrays = [counts_by_label[l].copy().astype(np.float64) for l in labels]

    if zero_outer_border:
        for a in arrays:
            a[0, :] = 0
            a[-1, :] = 0
            a[:, 0] = 0
            a[:, -1] = 0

    if log_scale:
        arrays = [np.log1p(a) for a in arrays]

    if per_panel_normalise:
        arrays = [(a / a.max()) if a.max() > 0 else a for a in arrays]
        vmin, vmax = 0.0, 1.0
    else:
        if shared_colour_scale:
            vmin = 0.0
            vmax = max(a.max() for a in arrays)
        else:
            vmin = vmax = None

    n = len(arrays)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, label, data in zip(axes, labels, arrays):
        ax.imshow(data, origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def coverage(counts: np.ndarray) -> int:
    """Number of states visited at least once."""
    return int((counts > 0).sum())


if __name__ == "__main__":
    import sys

    # Only run when not in a notebook.
    if not hasattr(sys, "ps1"):
        env = GridWorldEnv()
        s = env.reset()
        done = False
        while not done:
            env.render()
            a = np.random.choice([0, 1, 2, 3])
            s, r, done = env.step(a)
        env.render()
        print("Goal reached!")
