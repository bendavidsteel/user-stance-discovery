import collections
import json
import os

import matplotlib.pyplot as plt
import networkx as nx

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    graph_path = os.path.join(data_dir_path, 'graph_data.json')
    with open(graph_path, 'r') as f:
        node_link_data = json.load(f)

    multi_graph = nx.node_link_graph(node_link_data)
    multi_graph = nx.MultiGraph(multi_graph)
    multi_graph.remove_edges_from(nx.selfloop_edges(multi_graph))

    edge_filters = {
        'all': lambda edge_type: True,
        'video_comment': lambda edge_type: edge_type == 'video_comment',
        'comment_mention': lambda edge_type: edge_type == 'comment_mention',
        'video_share': lambda edge_type: edge_type == 'video_share',
        'video_mention': lambda edge_type: edge_type == 'video_mention',
        'comment_reply': lambda edge_type: edge_type == 'comment_reply'
    }

    fig, axes = plt.subplots(nrows=1, ncols=len(edge_filters), sharey=True, figsize=(20, 4))

    for ax, (name, edge_filter) in zip(axes, edge_filters.items()):
        graph = nx.MultiGraph(((u,v) for (u,v,d) in multi_graph.edges(data=True) if edge_filter(d['type'])))

        degree_freq = nx.degree_histogram(graph)

        ax.scatter(list(range(len(degree_freq))), degree_freq)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(name)

    fig.tight_layout()

    fig_dir_path = os.path.join(root_dir_path, 'figs')
    fig.savefig(os.path.join(fig_dir_path, 'degree_distributions.png'))


if __name__ == '__main__':
    main()