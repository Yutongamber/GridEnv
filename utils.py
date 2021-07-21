import matplotlib.collections as mcoll
import matplotlib.path as mpa
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


map_from_action_to_subplot = lambda a: (2, 6, 8, 4)[a]
map_from_action_to_name = lambda a: ("up", "right", "down", "left")[a]


def plot_values(grid, values, colormap='pink', vmin=0, vmax=10):
    plt.imshow(values - 1000 * (grid < 0), interpolation="nearest", cmap=colormap, vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks([])
    plt.colorbar(ticks=[vmin, vmax])


def plot_action_values(algo, grid, action_values, vmin=-5, vmax=5):
    q = action_values
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    # for a in [0, 1, 2, 3]:
    #     plt.subplot(4, 3, map_from_action_to_subplot(a))
    #     plot_values(grid, q[..., a], vmin=vmin, vmax=vmax)
    #     action_name = map_from_action_to_name(a)
    #     plt.title(r"$q(s, \mathrm{" + action_name + r"})$")
    #
    # plt.subplot(4, 3, 5)
    # v = np.max(q, axis=-1)
    # plot_values(grid, v, colormap='summer', vmin=vmin, vmax=vmax)
    # plt.title("$v(s)$")

    # Plot arrows:
    # plt.subplot(4, 3, 11)
    plot_values(grid, grid == 0, vmax=1)
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 0:
                argmax_a = np.argmax(q[row, col])
                if argmax_a == 0:
                    x = col
                    y = row + 0.5
                    dx = 0
                    dy = -0.8
                if argmax_a == 1:
                    x = col - 0.5
                    y = row
                    dx = 0.8
                    dy = 0
                if argmax_a == 2:
                    x = col
                    y = row - 0.5
                    dx = 0
                    dy = 0.8
                if argmax_a == 3:
                    x = col + 0.5
                    y = row
                    dx = -0.8
                    dy = 0
                plt.arrow(x, y, dx, dy, width=0.02, head_width=0.4, head_length=0.4, length_includes_head=True, fc='k',
                          ec='k')
    plt.savefig("./assets/" + "grid_" + algo + ".png")
    plt.show()


def plot_rewards(xs, rewards, color):
    mean = np.mean(rewards, axis=0)
    p90 = np.percentile(rewards, 90, axis=0)
    p10 = np.percentile(rewards, 10, axis=0)
    plt.plot(xs, mean, color=color, alpha=0.6)
    plt.fill_between(xs, p90, p10, color=color, alpha=0.3)


def parameter_study(parameter_values, parameter_name,
                    agent_constructor, env_constructor, color, repetitions=10, number_of_steps=int(1e4)):
    mean_rewards = np.zeros((repetitions, len(parameter_values)))
    greedy_rewards = np.zeros((repetitions, len(parameter_values)))
    for rep in range(repetitions):
        for i, p in enumerate(parameter_values):
            env = env_constructor()
            agent = agent_constructor()
            if 'eps' in parameter_name:
                agent.set_epsilon(p)
            elif 'alpha' in parameter_name:
                agent._step_size = p
            else:
                raise NameError("Unknown parameter_name: {}".format(parameter_name))
            mean_rewards[rep, i] = run_experiment(grid, agent, number_of_steps)
            agent.set_epsilon(0.)
            agent._step_size = 0.
            greedy_rewards[rep, i] = run_experiment(grid, agent, number_of_steps // 10)
            del env
            del agent

    plt.subplot(1, 2, 1)
    plot_rewards(parameter_values, mean_rewards, color)
    plt.yticks = ([0, 1], [0, 1])
    plt.ylabel("Average reward over first {} steps".format(number_of_steps), size=12)
    plt.xlabel(parameter_name, size=12)

    plt.subplot(1, 2, 2)
    plot_rewards(parameter_values, greedy_rewards, color)
    plt.yticks = ([0, 1], [0, 1])
    plt.ylabel("Final rewards, with greedy policy".format(number_of_steps), size=12)
    plt.xlabel(parameter_name, size=12)

def epsilon_greedy(q_values, epsilon):
  if epsilon < np.random.random():
    return np.argmax(q_values)
  else:
    return np.random.randint(np.array(q_values).shape[-1])


def colorline(x, y, z):
    """
    Based on:
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=plt.get_cmap('copper_r'),
                              norm=plt.Normalize(0.0, 1.0), linewidth=3)

    ax = plt.gca()
    ax.add_collection(lc)
    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plotting_helper_function(_x, _y, title=None, ylabel=None):
  z = np.linspace(0, 0.9, len(_x))**0.7
  colorline(_x, _y, z)
  plt.plot(0, 0, '*', color='#000000', ms=20, alpha=0.7, label='$w^*$')
  plt.plot(1, 1, '.', color='#ee0000', alpha=0.7, ms=20, label='$w_0$')
  min_y, max_y = np.min(_y), np.max(_y)
  min_x, max_x = np.min(_x), np.max(_x)
  min_y, max_y = np.min([0, min_y]), np.max([0, max_y])
  min_x, max_x = np.min([0, min_x]), np.max([0, max_x])
  range_y = max_y - min_y
  range_x = max_x - min_x
  max_range = np.max([range_y, range_x])
  plt.arrow(_x[-3], _y[-3], _x[-1] - _x[-3], _y[-1] - _y[-3], color='k',
            head_width=0.04*max_range, head_length=0.04*max_range,
            head_starts_at_zero=False)
  plt.ylim(min_y - 0.2*range_y, max_y + 0.2*range_y)
  plt.xlim(min_x - 0.2*range_x, max_x + 0.2*range_x)
  ax = plt.gca()
  ax.ticklabel_format(style='plain', useMathText=True)
  plt.legend(loc=2)
  plt.xticks(rotation=12, fontsize=10)
  plt.yticks(rotation=12, fontsize=10)
  plt.locator_params(nbins=3)
  if title is not None:
    plt.title(title, fontsize=20)
  if ylabel is not None:
    plt.ylabel(ylabel, fontsize=20)