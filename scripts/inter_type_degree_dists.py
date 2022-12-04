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
        'All': lambda edge_type: True,
        'Video Comments': lambda edge_type: edge_type == 'video_comment',
        'Comment Mentions': lambda edge_type: edge_type == 'comment_mention',
        'Video Shares': lambda edge_type: edge_type == 'video_share',
        'Video Mentions': lambda edge_type: edge_type == 'video_mention',
        'Comment Replies': lambda edge_type: edge_type == 'comment_reply'
    }

    fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(10, 5))

    for ax, (name, edge_filter) in zip(axes.ravel().tolist(), edge_filters.items()):
        graph = nx.MultiGraph(((u,v) for (u,v,d) in multi_graph.edges(data=True) if edge_filter(d['type'])))

        degree_freq = nx.degree_histogram(graph)

        ax.scatter(list(range(len(degree_freq))), degree_freq, s=5)
        ax.set_xlabel('Degree')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(name)

        if len(degree_freq) < 10:
            ax.set_xlim(left=0.8, right=11)

    axes[0][0].set_ylabel('Frequency')
    axes[1][0].set_ylabel('Frequency')
    fig.tight_layout()

    fig_dir_path = os.path.join(root_dir_path, 'figs')
    fig.savefig(os.path.join(fig_dir_path, 'degree_distributions.png'))


if __name__ == '__main__':
    main()