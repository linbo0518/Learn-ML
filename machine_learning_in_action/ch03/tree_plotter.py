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


def create_plot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(x_ticks=[], y_ticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_width = float(get_leafs_num(tree))
    plot_tree.total_depth = float(get_tree_depth(tree))
    plot_tree.x_coord = -0.5 / plot_tree.total_width
    plot_tree.y_coord = 1.0
    plot_tree(tree, (0.5, 1.0), '')
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


def plot_mid_text(center_coord, parent_coord, text_str):
    x_mid = (parent_coord[0] - center_coord[0]) / 2.0 + center_coord[0]
    y_mid = (parent_coord[1] - center_coord[1]) / 2.0 + center_coord[1]
    create_plot.ax1.text(x_mid, y_mid, text_str)


def plot_tree(decision_tree, parent_coord, node_text):
    leafs_num = get_leafs_num(decision_tree)
    tree_depth = get_tree_depth(decision_tree)
    first_str = list(decision_tree.keys())[0]
    center_coord = (plot_tree.x_coord +
                    (1.0 + float(leafs_num)) / 2.0 / plot_tree.total_width,
                    plot_tree.y_coord)
    plot_tree(center_coord, parent_coord, node_text)
    plot_node(first_str, center_coord, parent_coord, DECISION_NODE)
    second_dict = decision_tree[first_str]
    plot_tree.y_coord = plot_tree.y_coord - 1.0 / plot_tree.total_depth
    for each_key in list(second_dict.keys()):
        if type(second_dict[each_key]).__name__ == 'dict':
            plot_tree(second_dict[each_key], center_coord, str(each_key))
        else:
            plot_tree.x_coord = plot_tree.x_coord + 1.0 / plot_tree.total_width
            plot_node(second_dict[each_key],
                      (plot_tree.x_coord, plot_tree.y_coord), center_coord,
                      LEAF_NODE)
            plot_mid_text((plot_tree.x_coord, plot_tree.y_coord), center_coord,
                          str(each_key))
    plot_tree.y_coord = plot_tree.y_coord + 1.0 / plot_tree / plot_tree.total_depth
