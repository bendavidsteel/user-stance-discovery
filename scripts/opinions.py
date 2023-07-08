import torch

def degroot(user_state, new_content, attention):
    return user_state + torch.bmm(attention.unsqueeze(1), new_content).squeeze(1)

def friedkin_johnsen(original_user_state, content, stubbornness):
    return ((1 - stubbornness) * original_user_state) + (stubbornness * torch.sum(content))

def bounded_confidence(user_state, new_content, confidence_interval):
    diff = user_state - new_content
    mask = torch.norm(torch.abs(diff), dim=1) > confidence_interval
    return user_state + ((1 / torch.sum(mask)) * torch.sum(mask * diff))

def stochastic_bounded_confidence_categorical(user_state, new_content, content_mask, exponent):
    # compares prob of interaction between a number of people
    unsq_content_mask = content_mask.unsqueeze(2)
    # we can't use cdist because we can't mask in cdist, and unmasked cdist would be computationally inefficient
    content_dist = torch.cdist(user_state.unsqueeze(1), new_content * unsq_content_mask).squeeze(1) * content_mask

    vec_mask = content_mask.any(dim=1)

    # TODO use MaskedTensor
    numerator = torch.pow(content_dist + ((1 - content_mask) * 1), -1 * exponent.unsqueeze(1)) * content_mask
    probs = torch.zeros(numerator.shape, dtype=torch.float64)
    probs[vec_mask] = numerator[vec_mask] / torch.sum(numerator, dim=1, keepdims=True)[vec_mask]

    return probs

def stochastic_bounded_confidence_bernoulli(user_state, new_content, exponent, n, avg_content):
    # compares prob of interaction between a number of people
    numerator = torch.pow(torch.abs(user_state - new_content), -1 * exponent)
    return numerator / (numerator + ((n-1) * torch.pow(torch.abs(user_state - avg_content), -1 * exponent)))
