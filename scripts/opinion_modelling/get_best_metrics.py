import os

import opinion_datasets

def main():
    this_dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir_path = os.path.join(this_dir_path, "..", "..")

    dataset_name = "reddit"
    aggregation = "weighted_mean"

    if dataset_name == "reddit":
        this_dir_path = os.path.dirname(os.path.realpath(__file__))
        root_dir_path = os.path.join(this_dir_path, "..", "..")
        experi = "1sub_1year"
        if experi == "1sub_1year":
            topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "1sub_1year", "topics_minilm_0_2")
            fig_path = os.path.join(root_dir_path, "figs", "reddit", "1sub_1year")
        elif experi == "4sub_1year":
            topics_dir_path = os.path.join(root_dir_path, "data", "reddit", "4sub_1year")
            fig_path = os.path.join(root_dir_path, "figs", "reddit", "4sub_1year")
        dataset = opinion_datasets.RedditOpinionTimelineDataset(topics_dir_path, aggregation=aggregation)

    stance_precisions = []
    stance_recalls = []
    stance_f1s = []
    for stance_name in dataset.all_classifier_profiles:
        classifier_profiles = dataset.all_classifier_profiles[stance_name]
        best_favor_profile = None
        best_favor_f1 = 0
        best_against_profile = None
        best_against_f1 = 0
        for classifier_idx, classifier_profile in classifier_profiles.items():
            if classifier_profile["favor"]["f1"] > best_favor_f1:
                best_favor_profile = classifier_profile
                best_favor_f1 = classifier_profile["favor"]["f1"]
            if classifier_profile["against"]["f1"] > best_against_f1:
                best_against_profile = classifier_profile
                best_against_f1 = classifier_profile["against"]["f1"]

        stance_macro_precision = (best_favor_profile["favor"]["precision"] + best_against_profile["against"]["precision"]) / 2
        stance_macro_recall = (best_favor_profile["favor"]["recall"] + best_against_profile["against"]["recall"]) / 2
        stance_macro_f1 = (best_favor_profile["favor"]["f1"] + best_against_profile["against"]["f1"]) / 2

        stance_precisions.append(stance_macro_precision)
        stance_recalls.append(stance_macro_recall)
        stance_f1s.append(stance_macro_f1)

    avg_precision = sum(stance_precisions) / len(stance_precisions)
    avg_recall = sum(stance_recalls) / len(stance_recalls)
    avg_f1 = sum(stance_f1s) / len(stance_f1s)
    print(f"Average precision: {avg_precision}")
    print(f"Average recall: {avg_recall}")
    print(f"Average f1: {avg_f1}")

if __name__ == '__main__':
    main()