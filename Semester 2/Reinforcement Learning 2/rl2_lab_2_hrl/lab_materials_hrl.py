import textwrap
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List, Callable, Union, Optional, Protocol, runtime_checkable

import numpy as np
import matplotlib.pyplot as plt

State = Tuple[int, int]  # (y, x)
Action = int


class GridWorldEnv(object):
    def __init__(
        self,
        layout="""
        #############
        #......#....#
        #......#....#
        #...........#
        #......#....#
        #......#....#
        ###.######.##
        #.....#.....#
        #.....#.....#
        #...........#
        #.....#.....#
        #.....#.....#
        #############
        """,
        start_pos: State = (2, 2),
        goal_pos: State = (10, 10),
    ):
        # Parse the layout into a 2D numpy array.
        dedented = textwrap.dedent(layout)
        self.layout = np.array([list(line) for line in dedented.strip().splitlines()])
        self.height, self.width = self.layout.shape

        # Store start and goal positions.
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        # Initialise agent position.
        self.agent_pos: Optional[State] = None

        # Define the number of actions: East, South, West, North.
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
        action_effects = {
            0: (0, 1),  # East
            1: (1, 0),  # South
            2: (0, -1),  # West
            3: (-1, 0),  # North
        }

        # Get the effect of the action.
        if action in action_effects:
            dy, dx = action_effects[action]
            assert self.agent_pos is not None
            new_pos = (self.agent_pos[0] + dy, self.agent_pos[1] + dx)

            # Check for wall collisions.
            if self.layout[new_pos] != "#":
                self.agent_pos = new_pos

        # Check if the goal is reached.
        done = self.agent_pos == self.goal_pos
        reward = 1.0 if done else -0.01

        assert self.agent_pos is not None
        return self.agent_pos, reward, done

    def render(self):
        h, w = self.layout.shape
        img = np.ones((h, w, 3), dtype=float)

        walls = self.layout == "#"
        img[walls] = [0.0, 0.0, 0.0]

        if self.start_pos is not None:
            img[self.start_pos] = [0.0, 1.0, 0.0]
        if self.goal_pos is not None:
            img[self.goal_pos] = [1.0, 0.0, 0.0]
        if self.agent_pos is not None:
            img[self.agent_pos] = [0.0, 0.0, 1.0]

        if self._fig is None or self._ax is None or self._im is None:
            self._fig, self._ax = plt.subplots()
            self._im = self._ax.imshow(img, interpolation="nearest")
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._fig.tight_layout()
        else:
            self._im.set_data(img)

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
# Options
# ============================================================

PrimitiveOrOption = Union[Action, "Option"]


class Option:
    """
    A minimal Option interface.

    - initiation(s): Whether the option may be initiated in state s.
    - policy(s): Returns a primitive action to take in state s while executing the option.
    - termination(s): The probability of terminating upon arriving in state s.
    """

    def __init__(self, name: str):
        self.name = name

    def initiation(self, state: State) -> bool:
        raise NotImplementedError()

    def policy(self, state: State) -> Action:
        raise NotImplementedError()

    def termination(self, state: State) -> float:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"Option({self.name})"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.name))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Option) and self.__class__ is other.__class__ and self.name == other.name


def _step_deterministic(env: GridWorldEnv, state: State, action: Action) -> State:
    """
    Deterministic transition function matching env.step(...), but without rewards.

    Note: We assume the outer border of the grid is composed of walls, so indexing is safe.
    """
    y, x = state

    if action == 0:
        next_state = (y, x + 1)
    elif action == 1:
        next_state = (y + 1, x)
    elif action == 2:
        next_state = (y, x - 1)
    elif action == 3:
        next_state = (y - 1, x)
    else:
        raise ValueError(f"Invalid action {action}")

    if env.layout[next_state] == "#":
        return (y, x)
    return next_state


def shortest_path_distances(env: GridWorldEnv, start_state: State) -> Dict[State, int]:
    """
    Shortest-path distances on the grid (4-neighbour moves), respecting walls.

    Returns a dict mapping each reachable walkable state to its distance from `start_state`.
    """
    from collections import deque

    if env.layout[start_state] == "#":
        return {}

    distances: Dict[State, int] = {start_state: 0}
    q = deque([start_state])

    while q:
        state = q.popleft()
        y, x = state
        d = distances[state]

        # (dy, dx) for 0:E, 1:S, 2:W, 3:N
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_state = (y + dy, x + dx)

            if env.layout[next_state] == "#":
                continue
            if next_state in distances:
                continue

            distances[next_state] = d + 1
            q.append(next_state)

    return distances


class SubgoalOption(Option):
    """
    Option that navigates to a specific subgoal state.

    Construction:
      SubgoalOption(env, subgoal=(y, x), initiation_set_size=k)

    - Initiation set: the k nearest walkable states to the subgoal state, according to shortest path distance.
    - Termination: terminates upon reaching the subgoal OR leaving the initiation set.
    - Policy: computed once using value iteration with reward -1 per step, 0 on entering the subgoal.
    """

    def __init__(self, env: GridWorldEnv, subgoal: State, initiation_set_size: int, gamma: float = 1.0):
        super().__init__(name=f"SubgoalOption{subgoal}")
        self.subgoal = subgoal
        self.initiation_set_size = initiation_set_size
        self.gamma = gamma

        # Walkable states
        walkable: List[State] = []
        for y in range(env.height):
            for x in range(env.width):
                if env.layout[y, x] != "#":
                    walkable.append((y, x))

        dist = shortest_path_distances(env, subgoal)

        reachable = list(dist.keys())

        reachable_sorted = sorted(
            reachable,
            key=lambda state: (dist[state], state[0], state[1]),  # deterministic tie-break
        )

        self.initiation_set = set(reachable_sorted[:initiation_set_size])

        # Value iteration to compute the option's internal policy.
        V: Dict[State, float] = {state: 0.0 for state in walkable}
        theta = 1e-6
        actions = list(range(env.num_actions))

        while True:
            delta = 0.0
            for state in walkable:
                if state == subgoal:
                    continue

                best_value = -float("inf")
                for action in actions:
                    next_state = _step_deterministic(env, state, action)
                    reward = 0.0 if next_state == subgoal else -1.0
                    best_value = max(best_value, reward + self.gamma * V[next_state])

                old = V[state]
                V[state] = best_value
                delta = max(delta, abs(old - best_value))

            if delta < theta:
                break

        self._policy_table: Dict[State, Action] = {}
        for state in walkable:
            if state == subgoal:
                continue

            best_action = 0
            best_value = -float("inf")
            for action in actions:
                next_state = _step_deterministic(env, state, action)
                reward = 0.0 if next_state == subgoal else -1.0
                value = reward + self.gamma * V[next_state]
                if value > best_value:
                    best_value = value
                    best_action = action

            self._policy_table[state] = best_action

    def initiation(self, state: State) -> bool:
        return (state in self.initiation_set) and (state != self.subgoal)

    def termination(self, state: State) -> float:
        return float(state == self.subgoal or state not in self.initiation_set)

    def policy(self, state: State) -> Action:
        if state not in self._policy_table:
            raise KeyError(f"Option policy undefined at state={state}.")
        return self._policy_table[state]

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.subgoal, self.initiation_set_size))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SubgoalOption)
            and self.subgoal == other.subgoal
            and self.initiation_set_size == other.initiation_set_size
        )


# ============================================================
# Training + evaluation helpers
# ============================================================


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@runtime_checkable
class HRLAgent(Protocol):
    """
    Common public interface expected by the evaluation framework.

    Both the primitive-action agent and the options agent should implement this.
    """

    gamma: float

    def reset_agent_state(self) -> None: ...
    def train_step(self, env: GridWorldEnv, state: State) -> State: ...
    def greedy_action(self, state: State) -> PrimitiveOrOption: ...
    def select_action(self, state: State) -> PrimitiveOrOption: ...


@dataclass
class EvalPoint:
    step: int
    mean_return: float
    mean_length: float


def _run_greedy_episode(env: GridWorldEnv, agent: HRLAgent, max_steps_safety: int = 2_000) -> Tuple[float, int]:
    """
    Run ONE greedy evaluation episode (no learning) and return (discounted_return, length).

    If the agent selects an option, we execute the option until it terminates (or the env terminates).
    Termination is treated as a probability: terminate if U < termination(state).
    """
    agent.reset_agent_state()
    state = env.reset()

    discounted_return = 0.0
    discount = 1.0
    length = 0

    while True:
        choice = agent.greedy_action(state)

        if isinstance(choice, Option):
            option = choice
            while True:
                action = option.policy(state)
                next_state, reward, done = env.step(action)

                discounted_return += discount * float(reward)
                discount *= agent.gamma
                length += 1
                state = next_state

                if done or (random.random() < option.termination(state)):
                    break
                if length >= max_steps_safety:
                    break
        else:
            action = int(choice)
            next_state, reward, done = env.step(action)

            discounted_return += discount * float(reward)
            discount *= agent.gamma
            length += 1
            state = next_state

        if done or length >= max_steps_safety:
            break

    return discounted_return, length


def evaluate_greedy_policy(
    make_env: Callable[[], GridWorldEnv],
    agent: HRLAgent,
    *,
    num_episodes: int = 25,
) -> Tuple[float, float]:
    returns: List[float] = []
    lengths: List[int] = []

    for _ in range(num_episodes):
        env = make_env()
        G, T = _run_greedy_episode(env, agent)
        returns.append(G)
        lengths.append(T)

    return float(np.mean(returns)), float(np.mean(lengths))


def train_with_periodic_eval(
    make_env: Callable[[], GridWorldEnv],
    agent: HRLAgent,
    *,
    total_steps: int,
    eval_every: int,
    eval_episodes: int = 25,
) -> List[EvalPoint]:
    """
    Train the agent one environment step at a time, and periodically evaluate the greedy policy.

    Note: Evaluation uses a safety cap to avoid infinite loops. Training does not.
    """
    env = make_env()
    agent.reset_agent_state()
    state = env.reset()

    out: List[EvalPoint] = []
    next_eval = eval_every

    for step in range(1, total_steps + 1):
        state = agent.train_step(env, state)

        if step == next_eval:
            mean_ret, mean_len = evaluate_greedy_policy(make_env, agent, num_episodes=eval_episodes)
            out.append(EvalPoint(step=step, mean_return=mean_ret, mean_length=mean_len))
            next_eval += eval_every

    return out


def run_experiment(
    make_env: Callable[[], GridWorldEnv],
    make_agent: Callable[[GridWorldEnv], HRLAgent],
    *,
    total_steps: int,
    eval_every: int,
    eval_episodes: int = 25,
    num_runs: int = 5,
    base_seed: int = 0,
) -> List[List[EvalPoint]]:
    """
    Run multiple seeds and return a list of evaluation traces.
    """
    traces: List[List[EvalPoint]] = []

    for i in range(num_runs):
        set_global_seeds(base_seed + i)
        env0 = make_env()
        agent = make_agent(env0)
        traces.append(
            train_with_periodic_eval(
                make_env, agent, total_steps=total_steps, eval_every=eval_every, eval_episodes=eval_episodes
            )
        )

    return traces


def _mean_and_ci95(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    xs: shape (num_runs, num_points)
    Returns (mean, lo, hi) using normal approx CI: mean ± 1.96 * (std / sqrt(n))
    """
    mean = xs.mean(axis=0)
    if xs.shape[0] == 1:
        return mean, mean, mean
    std = xs.std(axis=0, ddof=1)
    se = std / np.sqrt(xs.shape[0])
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    return mean, lo, hi


def plot_learning_curves(
    traces_by_label: Dict[str, List[List[EvalPoint]]],
) -> None:
    """
    Plot mean learning curves with 95% confidence intervals.

    traces_by_label[label] is a list of runs, each run is a list of EvalPoint.
    """
    plt.figure()

    for label, runs in traces_by_label.items():
        steps = np.array([p.step for p in runs[0]], dtype=np.int64)
        returns = np.array([[p.mean_return for p in run] for run in runs], dtype=np.float64)

        mean, lo, hi = _mean_and_ci95(returns)
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, lo, hi, alpha=0.2)

    plt.title("Episodic Return Earned During Evaluation Episode")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episodic Return")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grid_with_coordinates(env: GridWorldEnv) -> None:
    """
    Plot the grid layout with (y, x) coordinates written on each walkable tile.

    - Walls are black and have no text.
    - Walkable cells are white with black coordinate text.
    - Does not display the agent position.
    """
    height, width = env.layout.shape

    img = np.ones((height, width, 3), dtype=float)
    walls = env.layout == "#"
    img[walls] = [0.0, 0.0, 0.0]

    fig, ax = plt.subplots()
    ax.imshow(img, interpolation="nearest")

    for y in range(height):
        for x in range(width):
            if env.layout[y, x] != "#":
                ax.text(
                    x,
                    y,
                    f"({y},{x})",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )

    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.show()
