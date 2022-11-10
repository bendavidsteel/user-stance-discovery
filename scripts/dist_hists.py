import json
import os

import matplotlib.pyplot as plt

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    user_seq_path = os.path.join(data_dir_path, 'user_seqs.json')
    with open(user_seq_path, 'r') as f:
        user_seqs = json.load(f)

    viewed_dists = []
    global_dists = []
    sec_ord_dists = []
    for user_id, user_seq in user_seqs.items():
        for inter in user_seq:
            if 'viewed_content_dist_share' in inter:
                viewed_dists.append(inter['viewed_content_dist_share'])
            if 'global_content_dist_shares' in inter:
                global_dists += inter['global_content_dist_shares']
            if 'second_order_neighbour_content_dist_shares' in inter:
                sec_ord_dists += inter['second_order_neighbour_content_dist_shares']

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(viewed_dists, bins=100, density=True, alpha=0.5, label='Viewed')
    ax.hist(global_dists, bins=100, density=True, alpha=0.5, label='Global')

    ax.set_ylabel('Density')
    ax.set_xlabel('Distance')

    fig.legend()

    fig_path = os.path.join(root_dir_path, 'figs', 'dist_hists.png')
    fig.savefig(fig_path)


if __name__ == '__main__':
    main()