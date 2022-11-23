import os

import umap
import numpy as np

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_path = os.path.join(this_dir_path, '..', 'data')

    video_embeddings_cache_path = os.path.join(data_dir_path, 'all_video_desc_bertweet_embeddings.npy')
    with open(video_embeddings_cache_path, 'rb') as f:
            video_embeddings = np.load(f)

    comment_embeddings_cache_path = os.path.join(data_dir_path, 'all_english_comment_bertweet_embeddings.npy')
    with open(comment_embeddings_cache_path, 'rb') as f:
            comment_embeddings = np.load(f)

    # sample some embeddings
    sample_percent = 0.33
    video_sample_idx = np.random.choice(video_embeddings.shape[0], round(sample_percent * video_embeddings.shape[0]))
    comment_sample_idx = np.random.choice(comment_embeddings.shape[0], round(sample_percent * comment_embeddings.shape[0]))
    video_sample_embeddings = video_embeddings[video_sample_idx, :]
    comment_sample_embeddings = comment_embeddings[comment_sample_idx, :]
    sample_embeddings = np.concatenate((video_sample_embeddings, comment_sample_embeddings), axis=0)

    embeddings = np.concatenate((video_embeddings, comment_embeddings), axis=0)
    
    low_memory = False
    # using same settings as bertopic
    umap_model = umap.UMAP(n_neighbors=15,
                           n_components=5,
                           min_dist=0.0,
                           metric='cosine',
                           low_memory=low_memory,
                           verbose=True)

    umap_model.fit(sample_embeddings)
    umap_embeddings = umap_model.transform(embeddings)
    umap_embeddings = np.nan_to_num(umap_embeddings)

    video_umap_embeddings = umap_embeddings[:video_embeddings.shape[0], :]
    comment_umap_embeddings = umap_embeddings[-comment_embeddings.shape[0]:, :]

    comment_umap_embeddings_cache_path = os.path.join(data_dir_path, 'all_english_comment_bertweet_umap_embeddings.npy')
    video_umap_embeddings_cache_path = os.path.join(data_dir_path, 'all_video_desc_bertweet_umap_embeddings.npy')

    with open(comment_umap_embeddings_cache_path, 'wb') as f:
        np.save(f, comment_umap_embeddings)

    with open(video_umap_embeddings_cache_path, 'wb') as f:
        np.save(f, video_umap_embeddings)

if __name__ == '__main__':
    main()