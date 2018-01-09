import matplotlib.pyplot as plt

DECISION_NODE = dict(boxstyle="sawtooth", fc="0.8")
LEAF_NODE = dict(boxstyle="round4", fc="0.8")
ARROW_ARGS = dict(arrowstyle="<-")


def plot_node(node_text, center_coord, parent_coord, node_type):
    create_plot.ax1.annotate(
        node_text,
        xy=parent_coord,
        xycoords='axes fraction',
        xytext=center_coord,
        textcoords='axes fraction',
        va="center",
        ha="center",
        bbox=node_type,
        arrowprops=ARROW_ARGS)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), DECISION_NODE)
    plot_node('a leaf node', (0.8, 0.1), (0.3, 0.8), LEAF_NODE)
    plt.show()