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


def get_leafs_num(decision_tree):
    leafs_num = 0
    first_str = list(decision_tree.keys())[0]
    second_dict = decision_tree[first_str]
    for each_key in list(second_dict.keys()):
        if type(second_dict[each_key]).__name__ == 'dict':
            leafs_num += get_leafs_num(second_dict[each_key])
        else:
            leafs_num += 1
    return leafs_num


def get_tree_depth(decision_tree):
    max_depth = 0
    first_str = list(decision_tree.keys())[0]
    second_dict = decision_tree[first_str]
    for each_key in list(second_dict.keys()):
        if type(second_dict[each_key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[each_key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth
