import collections
import json
import os

from tqdm import tqdm

import scripts.degree_hists.dist_hists as dist_hists

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    root_dir_path = os.path.join(this_dir_path, '..')
    data_dir_path = os.path.join(root_dir_path, 'data')

    change_type = 'changepoint'

    file_name = f'user_seqs_{change_type}.json'
    user_seq_path = os.path.join(data_dir_path, file_name)
    with open(user_seq_path, 'r') as f:
        user_seqs = json.load(f)

    edge_type = 'comment_reply'
    fig_type = 'dist'

    for user_id, user_seq in tqdm(user_seqs.items()):
        for inter in user_seq:

            moves = {}
            inter_type = inter['edge_data']['type']
            for key, value in inter.items():
                if 'euclid' in key or 'cosine' in key:
                    if fig_type == 'movement' and '_user_' in key and 'prev' not in key:
                        key_suffix = 19 if 'euclid' in key else 18
                        other_user_dist = value
                        prev_other_dist = inter[key.replace('user', 'prev')]
                        user_prev_dist = inter[f"prev_user_{key[-key_suffix:]}"]
                        moves[f"{key}_between_{inter_type}"] = dist_hists.between_metric(other_user_dist, prev_other_dist, user_prev_dist)
                        moves[f"{key}_away_{inter_type}"] = dist_hists.away_metric(other_user_dist, prev_other_dist, user_prev_dist)

            

if __name__ == '__main__':
    main()