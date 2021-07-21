from algo.tabularq import GeneralQ
from utils import *
from grid_world import Grid, GRID_LAYOUT
import argparse


# Agent settings.
# Do not modify this cell.
epsilon = 0.25
step_size = 0.1


def run_experiment(env, agent, number_of_steps):
    mean_reward = 0.
    try:
        action = agent.initial_action()
    except AttributeError:
        action = 0
    for i in range(number_of_steps):
        reward, discount, next_state = grid.step(action)
        action = agent.step(reward, discount, next_state)
        mean_reward += reward
    return mean_reward / float(number_of_steps)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default="q_learning", type=str, help="q_learning/sarsa/expected_sarsa/doubleq")
    args = parser.parse_args()

    if args.algo == "q_learning":
        def behaviour_policy(q):
            return epsilon_greedy(q, epsilon)

        def target_policy(q, a):
            return np.eye(len(q))[np.argmax(q)]

    if args.algo == "sarsa":
        def behaviour_policy(q):
            return epsilon_greedy(q, epsilon)

        def target_policy(q, a):
            return np.eye(len(q))[a]

    if args.algo == "expected_sarsa":
        def behaviour_policy(q):
            return epsilon_greedy(q, epsilon)

        def target_policy(q, a):
            greedy = np.eye(len(q))[np.argmax(q)]
            return greedy - greedy * epsilon + epsilon / 4

    if args.algo == "doubleq":
        def behaviour_policy(q):
            return epsilon_greedy(q, epsilon)

        def target_policy(q, a):
            max_q = np.max(q)
            pi = np.array([1. if qi == max_q else 0. for qi in q])
            return pi / sum(pi)

    learned_qs = []
    for _ in range(5):
        grid = Grid()
        agent = GeneralQ(grid._layout.size, 4, grid.get_obs(), target_policy,
                         behaviour_policy, double=False, step_size=step_size)
        run_experiment(grid, agent, int(1e5))
        learned_qs.append(agent.q_values.reshape(grid._layout.shape + (4,)))

    avg_qs = sum(learned_qs) / len(learned_qs)
    plot_action_values(args.algo, GRID_LAYOUT, avg_qs, vmin=-20, vmax=100)


