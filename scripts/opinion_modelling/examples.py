import os

import pandas as pd

import opinion_datasets

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    aggregation = "mean"
    dataset = opinion_datasets.RedditOpinionTimelineDataset(aggregation=aggregation)

    opinion_df = pd.DataFrame(dataset[0], columns=dataset.stance_columns)

    


if __name__ == "__main__":
    main()