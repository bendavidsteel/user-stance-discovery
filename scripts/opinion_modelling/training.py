import os

import numpy as np
import pandas as pd
import torch
import tqdm

from pyro.generic import infer, optim, pyro_backend, pyro

import datasets
import generative
import discriminative

def get_trainer(model_func, guide=None):
    if guide is None:
        guide = infer.autoguide.AutoDelta(model_func)
    adam = optim.ClippedAdam({"lr": 0.001})
    elbo = infer.Trace_ELBO()
    svi = infer.SVI(model_func, guide, adam, elbo)
    return svi

def training(dataloader, batch_size):

    backend = "pyro"
    with pyro_backend(backend):

        if backend == "pyro":
            pyro.enable_validation(True)
            pyro.set_rng_seed(1)

        discriminative_model = discriminative.SocialDiscrimativeModel()
        trainer = get_trainer(discriminative_model.reply_and_write_comment)#, guide=reply_and_write_comment_guide)

        num_epochs = 100

        for epoch in range(num_epochs):
            running_loss = 0.0
            num_preds = 0

            # Iterate over data.
            bar = tqdm.tqdm(
                dataloader,
                desc="Epoch {}".format(epoch),
            )
            for i, batch in enumerate(bar):
                padded_comments_opinions = batch['padded_comments_opinions']
                mask_comments_opinions = batch['mask_comments_opinions']
                reply_comment = batch['reply_comment']
                actual_comment = batch['actual_comment']

                loss = trainer.step(padded_comments_opinions, mask_comments_opinions, actual_reply_comment=reply_comment, actual_comment=actual_comment)

                # statistics
                running_loss += loss / batch_size
                num_preds += 1
                if i % 10 == 0:
                    bar.set_postfix(
                        loss="{:.2f}".format(running_loss / num_preds),
                    )

            epoch_loss = running_loss / len(dataloader.dataset)

        alpha_q = pyro.param("alpha_q").item()
        beta_q = pyro.param("beta_q").item()

def main():
    this_dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(this_dir_path, '..', 'data', 'bertweet_base_5_100_100')

    batch_size = 64
    num_workers = 0
    pin_memory = False
    shuffle = False

    dataset_type = 'generative'
    if dataset_type == 'generative':
        dataset = datasets.GenerativeDataset(generative.SocialGenerativeModel)
    elif dataset_type == 'reddit':
        dataset = datasets.RedditInteractionDataset(data_path)
    elif dataset_type == 'tiktok':
        dataset = datasets.TikTokInteractionDataset(data_path)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(dataset), batch_size=batch_size, drop_last=False
        ),
        batch_size=None,
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    training(dataloader, batch_size)

if __name__ == '__main__':
    main()
