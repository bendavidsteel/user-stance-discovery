import os

import torch

import pyro

import scripts.opinion_modelling.discriminative as discriminative

def test_choices_examples():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data', 'bertweet_base_5_100_100')

    batch_size = 1
    num_workers = 0
    pin_memory = False
    shuffle = True

    inter_dataset = discriminative.InteractionDataset(data_path)
    inter_dataloader = torch.utils.data.DataLoader(inter_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    pyro.enable_validation(True)
    pyro.set_rng_seed(1)

    trainer = discriminative.get_trainer(discriminative.choose_reply_comment)#, guide=choose_reply_comment_guide)

    example_seqs = [[0, 1]]

    # 14103
    # 10244
    # 5245

    for example_seq in example_seqs:
        running_loss = 0.
        for example_idx in example_seq:
            example = inter_dataset[example_idx]
            batch = inter_dataloader.collate_fn([example])

            user_state = batch['user_state']
            padded_comments_opinions = batch['padded_comments_opinions']
            mask_comments_opinions = batch['mask_comments_opinions']
            reply_comment = batch['reply_comment']

            loss = trainer.step(user_state, padded_comments_opinions, mask_comments_opinions, actual_reply_comment=reply_comment)
            running_loss += loss / batch_size

if __name__ == "__main__":
    test_choices_examples()