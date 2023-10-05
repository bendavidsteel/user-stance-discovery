import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
import pandas as pd
import torch
import tqdm

import opinions

class SocialPlatform:
    def __init__(self):
        self.reset()

    def reset(self):
        self.content_idx = 0
        self.posts = pd.DataFrame(columns=['position', 'id', 'time']).set_index('id')
        self.comments = pd.DataFrame(columns=['position', 'id', 'post', 'comment', 'time']).set_index('id')
        self.interactions = nx.DiGraph()

    def get_id(self):
        id = self.content_idx
        self.content_idx += 1
        return id

    def create_post(self, time_step, pos):
        id = self.get_id()
        new_post = {
            'position': pos,
            'time': time_step
        }
        self.posts = pd.concat([self.posts, pd.DataFrame([new_post], index=[id])])
        self.interactions.add_node(id)

    def create_comment(self, time_step, post_id, pos, comment_id=None):
        id = self.get_id()
        new_comment = {
            'position': pos,
            'post': post_id,
            'comment': comment_id,
            'time': time_step
        }
        self.comments = pd.concat([self.comments, pd.DataFrame([new_comment], index=[id])])
        self.interactions.add_edge(id, post_id)
        if comment_id is not None:
            self.interactions.add_edge(id, comment_id)

    def get_post_comments(self, post_ids, limit):
        all_comments = []
        for post_id in post_ids:
            # get predecessors
            comment_nodes = self.interactions[post_id]
            if len(comment_nodes) == 0:
                continue

            # get most recent comments
            comments = self.comments[self.comments['idx'].isin(comment_nodes)]
            comments = comments.sort_values(by='time', ascending=False)
            if limit is not None:
                comments = comments[:limit]
            all_comments.append(comments)

        return all_comments
    
    def get_post_comment_positions(self, post_ids, limit):
        comment_dfs = self.get_post_comments(post_ids, limit=limit)
        if len(comment_dfs) == 0:
            return torch.zeros(len(post_ids), limit, 2), torch.zeros(len(post_ids), limit), [{} for _ in range(len(post_ids))]
        
        comment_positions = torch.zeros(len(post_ids), limit, 2)
        comment_mask = torch.zeros(len(post_ids), limit)
        idx_to_ids = [{} for _ in range(len(post_ids))]
        for i, comments in enumerate(comment_dfs):
            comment_mask[i, :len(comments)] = 1
            if len(comments) > 0:
                comment_positions[i] = torch.stack([t for t in comments['position'].values])
                idx_to_ids[i] = {idx: id for idx, id in enumerate(comments.index.values)}

        return comment_positions, comment_mask, idx_to_ids
    
    def get_post_recommended_comment_positions(self, post_ids, limit=5):
        comments_dfs = self.get_post_comments(post_ids, limit=None)
        if len(comments_dfs) == 0:
            return torch.zeros(len(post_ids), limit, 2), torch.zeros(len(post_ids), limit), [{} for _ in range(len(post_ids))]
        
        comment_positions = torch.zeros(len(post_ids), limit, 2)
        comment_mask = torch.zeros(len(post_ids), limit)
        idx_to_ids = [{} for _ in range(len(post_ids))]
        for i, comments in enumerate(comments_dfs):
            comment_mask[i, :len(comments)] = 1
            if len(comments) > 0:
                recent_comments = comments.sort_values(by='time', ascending=False)
                comment_positions[i] = torch.stack([t for t in recent_comments['position'].values])
                idx_to_ids[i] = {idx: id for idx, id in enumerate(recent_comments.index.values)}

        return comment_positions, comment_mask, idx_to_ids

    
    def get_comments(self):
        return self.comments
    
    def get_posts(self):
        return self.posts
    
    def num_posts(self):
        return len(self.posts)
    
    def get_comments_positions(self):
        if len(self.comments) == 0:
            return torch.zeros(0, 2), {}
        idx_to_id = {idx: id for idx, id in enumerate(self.comments.index.values)}
        return torch.stack([t for t in self.comments['position'].values]), idx_to_id
    
    def get_posts_positions(self, ids=None):
        if len(self.posts) == 0:
            return torch.zeros(0, 2), {}
        if ids:
            posts = self.posts.loc[ids]
            # ensure order is kept
            assert all(post_id == id for post_id, id in zip(posts.index.values, ids))
        else:
            posts = self.posts
        idx_to_id = {idx: id for idx, id in enumerate(posts.index.values)}
        return torch.stack([t for t in posts['position'].values]), idx_to_id
    
    def get_recommended_posts_positions(self, limit=5):
        # TODO more complex recommendation algorithm
        posts = self.posts.sort_values(by='time', ascending=False)
        if limit is not None:
            posts = posts[:limit]
        return torch.stack([t for t in posts['position'].values]), {idx: id for idx, id in enumerate(posts.index.values)}

    def update(self, time_step):
        post_period = 10
        posts_to_delete = self.posts[self.posts['time'] < time_step - post_period]
        self.posts = self.posts.drop(posts_to_delete.index)
        self.interactions.remove_nodes_from(posts_to_delete.index.values)

        comment_period = 10
        comments_to_delete = self.comments[self.comments['time'] < time_step - comment_period]
        self.comments = self.comments.drop(comments_to_delete.index)
        self.interactions.remove_nodes_from(comments_to_delete.index.values)


class SocialGenerativeModel:
    def __init__(
            self, 
            num_users=1000, 
            sbc_exponent_loc=0., 
            sbc_exponent_scale=10., 
            sus_loc=0.5, 
            sus_scale=5.,
            seen_att_loc=0.1,
            seen_att_scale=0.1,
            reply_att_loc=0.2,
            reply_att_scale=0.1,
            post_att_loc=0.2,
            post_att_scale=0.1,
            content_scale=1.
        ):
        self.num_users = num_users

        self.users = pd.DataFrame(columns=['initial_position', 'sbc_exponent', 'sus', 'seen_att', 'reply_att', 'post_att'])

        self.sbc_exponent_loc = sbc_exponent_loc
        self.sbc_exponent_scale = sbc_exponent_scale

        self.sus_loc = sus_loc
        self.sus_scale = sus_scale

        # TODO replace all attention with an influence network
        self.seen_att_loc = seen_att_loc
        self.seen_att_scale = seen_att_scale

        self.reply_att_loc = reply_att_loc
        self.reply_att_scale = reply_att_scale

        self.post_att_loc = post_att_loc
        self.post_att_scale = post_att_scale

        self.content_scale = content_scale

        self.platform = SocialPlatform()

    def reset(self):
        self.reset_users()
        self.platform.reset()
        self.seed_posts()

    def reset_users(self):
        sbc_exponent_dist = torch.distributions.Normal(self.sbc_exponent_loc, self.sbc_exponent_scale)
        sus_dist = torch.distributions.Normal(self.sus_loc, self.sus_scale)
        seen_att_dist = torch.distributions.Normal(self.seen_att_loc, self.seen_att_scale)
        reply_att_dist = torch.distributions.Normal(self.reply_att_loc, self.reply_att_scale)
        post_att_dist = torch.distributions.Normal(self.post_att_loc, self.post_att_scale)

        user_dicts = []
        for i in range(self.num_users):
            initial_dist = torch.distributions.Uniform(-1, 1)
            initial_pos = initial_dist.sample((2,))

            sbc_exponent = sbc_exponent_dist.sample().type(dtype=torch.float32)
            sus = sus_dist.sample().clamp(0., 1.).type(dtype=torch.float32)
            seen_att = seen_att_dist.sample().clamp(0., 1.).type(dtype=torch.float32)
            reply_att = reply_att_dist.sample().clamp(0., 1.).type(dtype=torch.float32)
            post_att = post_att_dist.sample().clamp(0., 1.).type(dtype=torch.float32)

            user = {
                'initial_position': initial_pos,
                'position': initial_pos.clone(),
                'sbc_exponent': sbc_exponent,
                'sus': sus,
                'seen_attention': seen_att,
                'reply_attention': reply_att,
                'post_attention': post_att
            }
            user_dicts.append(user)

        self.users = pd.DataFrame(user_dicts)

    def seed_posts(self):
        self._create_posts(0, 1 / self.num_users)

    def get_users(self):
        return self.users
    
    def get_users_positions(self):
        if len(self.users) == 0:
            return torch.zeros(0, 2)
        return torch.stack([t for t in self.users['position'].values])

    def get_user_content(self):
        return self.get_users_tensor('content')

    def get_users_tensor(self, attr):
        return torch.stack([t for t in self.users[attr].values])

    def create_posts(self, time_step):
        self._create_posts(time_step, 1 / (self.num_users * 2))

    def _create_posts(self, time_step, prob_threshold):
        to_post = torch.distributions.Uniform(0., 1.).sample((self.num_users,)) < prob_threshold
        if not to_post.any():
            return
        users_to_post = self.users[to_post.numpy()]
        users_to_post_positions = torch.stack([pos for pos in users_to_post['position']])
        content_distributions = torch.distributions.MultivariateNormal(users_to_post_positions, self.content_scale * torch.eye(users_to_post_positions.shape[1]))
        post_contents = content_distributions.sample()
        [self.platform.create_post(time_step, post_content) for post_content in post_contents]

    def choose_post(self):
        if self.platform.num_posts() != 0:
            users_pos = self.get_users_positions()
            sbc_exponent = self.get_users_tensor('sbc_exponent')
            posts_pos, idx_to_id = self.platform.get_recommended_posts_positions()
            post_content = posts_pos.expand((self.num_users, posts_pos.shape[0], 2))
            post_content_mask = torch.ones((self.num_users, posts_pos.shape[0]), dtype=torch.float32)

            # TODO allow to not choose a post
            post_diff, chosen_posts = opinions.sbc_choice(
                users_pos,
                post_content,
                post_content_mask,
                sbc_exponent
            )

            chosen_post_ids = [idx_to_id[idx] for idx in chosen_posts.numpy()]

            return post_diff, chosen_post_ids
        
    def choose_comment(self, chosen_posts):
        # get post positions
        # TODO we are comparing post embeddings and comment embeddings here. this relies on their embeddings being aligned
        posts, idx_to_ids = self.platform.get_posts_positions(ids=chosen_posts)
        posts_mask = torch.ones(posts.shape[0], dtype=torch.float32)

        # see n comments on post content, sorted by time
        num_comments = 3
        comments, comments_mask, idx_to_ids = self.platform.get_post_recommended_comment_positions(chosen_posts, num_comments)

        users_pos = self.get_users_positions()
        seen_diff = torch.mean(comments_mask.unsqueeze(2) * (comments - users_pos.unsqueeze(1)), axis=1)
        seen_attention = self.get_users_tensor('seen_attention').unsqueeze(1)

        sbc_exponent = self.get_users_tensor('sbc_exponent')

        content = torch.cat([posts.unsqueeze(1), comments], dim=1)
        content_mask = torch.cat([posts_mask.unsqueeze(1), comments_mask], dim=1)

        reply_diff, chosen_content = opinions.sbc_choice(
            users_pos, 
            content, 
            content_mask, 
            sbc_exponent
        )

        reply_attention = self.get_users_tensor('reply_attention').unsqueeze(1)

        # if replying to post, reply diff should be zero
        reply_diff *= (chosen_content > 0).unsqueeze(1)

        comments_diff = seen_attention * seen_diff + reply_attention * reply_diff

        def get_id_from_idx(idx, idx_to_id):
            if idx == 0: # chosen a base comment, i.e. reply to post
                return None
            idx -= 1 # shift idx to account for post
            if idx in idx_to_id:
                return idx_to_id[idx]
            raise Exception('Invalid idx')

        chosen_content_ids = [get_id_from_idx(idx, idx_to_id) for idx_to_id, idx in zip(idx_to_ids, chosen_content.numpy())]

        return comments_diff, chosen_content_ids

    def update(self, time_step):
        # update the platform
        self.platform.update(time_step)

        # let users create posts
        self.create_posts(time_step)

        if self.platform.num_posts() == 0:
            return

        users_pos = self.get_users_positions()

        # TODO currently users choose from all posts, they should be restricted, either randomly, or by a recommendation algorithm, or by social influence
        # let users choose the post they want to interact with
        post_diff, chosen_post_ids = self.choose_post()
        post_attention = self.get_users_tensor('post_attention').unsqueeze(1)
        users_pos += post_attention * post_diff

        # let users choose the comments they want to interact with
        # TODO let social influence have an effect on comment choice too
        comments_diff, chosen_content_ids = self.choose_comment(chosen_post_ids)
        users_pos += comments_diff
    
        # update user positions
        sus = self.get_users_tensor('sus').view(-1, 1)
        initial_pos = self.get_users_tensor('initial_position')
        new_users_pos = (sus * users_pos) + ((1 - sus) * initial_pos)

        self.users['position'] = [pos for pos in new_users_pos]

        # create user comments
        comment_distributions = torch.distributions.MultivariateNormal(new_users_pos, self.content_scale * torch.eye(new_users_pos.shape[1]))
        comment_positions = comment_distributions.sample()

        [
            self.platform.create_comment(time_step, chosen_post_idx, comment_pos, comment_id=chosen_comment_idx)
            for chosen_post_idx, chosen_comment_idx, comment_pos 
            in zip(chosen_post_ids, chosen_content_ids, comment_positions)
        ]


def evaluate_user_polarization(users):
    # esteban ray
    
    return 0

def evaluate_content_polarization(posts, comments):
    # esteban ray
    return 0

def evaluate_user_movement(users):
    return 0

class UpdateDist:
    def __init__(self, social_context, user_ax, content_ax):
        self.social_context = social_context

        user_pos = self.social_context.get_users_positions()
        self.user_scatter = user_ax.scatter(user_pos[:, 0], user_pos[:, 1])
        self.user_ax = user_ax

        self.post_scatter = content_ax.scatter([], [], marker='x', color='red', zorder=10)
        self.comment_scatter = content_ax.scatter([], [], zorder=1)
        self.content_ax = content_ax

        # Set up plot parameters
        self.user_ax.set_xlim(-5, 5)
        self.user_ax.set_ylim(-5, 5)
        self.user_ax.grid(True)

        self.content_ax.set_xlim(-5, 5)
        self.content_ax.set_ylim(-5, 5)
        self.content_ax.grid(True)

    def __call__(self, i):
        # This way the plot can continuously run and we just keep
        # watching new realizations of the process
        if i == 0:
            self.social_context.reset()

        self.social_context.update(i)

        user_pos = self.social_context.get_users_positions()
        self.user_scatter.set_offsets(user_pos)

        comment_pos, _ = self.social_context.platform.get_comments_positions()
        self.comment_scatter.set_offsets(comment_pos)

        post_pos, _ = self.social_context.platform.get_posts_positions()
        self.post_scatter.set_offsets(post_pos)

        # TODO draw polarization
        user_polarization = evaluate_user_polarization(user_pos)
        content_polarization = evaluate_content_polarization(post_pos, comment_pos)
        user_movement = evaluate_user_movement(user_pos)
        
        return self.user_scatter, self.post_scatter, self.comment_scatter

def main():
    # Fixing random state for reproducibility
    torch.manual_seed(19680801)

    display = True

    num_steps = 100
    num_users = 100
    social_context = SocialGenerativeModel(num_users=num_users)
    if display:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 5))
        ud = UpdateDist(social_context, axes[0], axes[1])
        anim = FuncAnimation(fig, ud, frames=num_steps, interval=100, blit=True)
        plt.show()
    else:
        social_context.reset()
        for i in tqdm.tqdm(range(num_steps)):
            social_context.update(i)

if __name__ == '__main__':
    main()