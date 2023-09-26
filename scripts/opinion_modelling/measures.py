import torch

######################################################
## The Esteban-Ray polarization measure implementation
######################################################

def belief_2_dist(belief_vec, num_bins=10):
    """Takes a belief state `belief_vec` and discretizes it into a fixed
    number of bins.
    """
    # stores labels of bins
    # the value of a bin is the medium point of that bin
    bin_labels = [(i + 0.5)/num_bins for i in range(num_bins)]

    # stores the distribution of labels
    bin_prob = [0] * num_bins
    # for all agents...
    for belief in belief_vec:
        # computes the bin into which the agent's belief falls
        bin_ = math.floor(belief * num_bins)
        # treats the extreme case in which belief is 1, putting the result in the right bin.
        if bin_ == num_bins:
            bin_ = num_bins - 1
        # updates the frequency of that particular belief
        bin_prob[bin_] += 1 / len(belief_vec)
    # bundles into a matrix the bin_labels and bin_probabilities.
    dist = np.array([bin_labels,bin_prob])
    # returns the distribution.
    return dist

def pol_ER(dist, alpha=1.6, K=1):
    """Computes the Esteban-Ray polarization of a distribution.
    """
    # recover bin labels
    bin_labels = dist[0]
    # recover bin probabilities
    bin_prob = dist[1]

    diff = np.ones((len(bin_labels), 1)) @ bin_labels[np.newaxis]
    diff = np.abs(diff - np.transpose(diff))
    pol = (bin_prob ** (1 + alpha)) @ diff @ bin_prob
    # scales the measure by the constant K, and returns it.
    return K * pol


######################################################
## Gubanov et al. polarization measure implementation
######################################################

def symmetric_polarization(beliefs):
    """Computes the symmetric polarization of a group of agents.
    """
    assert beliefs.sum(axis=1).all() == 1, "Beliefs must sum to 1 along each row."

    n = beliefs.shape[0]
    # computes the mean belief of the group along each belief axis
    mean_belief = beliefs.mean(axis=0)
    # computes the polarization of the group
    polarization = (beliefs - mean_belief).abs().sum() / (2 * n)
    # returns the polarization.
    return polarization

def asymmetric_polarization(beliefs):
    '''Computes the asymmetric polarization of a group of agents.
    '''
    assert beliefs.sum(axis=1).all() == 1, "Beliefs must sum to 1 along each row."

    n = beliefs.shape[0]
    m = beliefs.shape[1]

    # find every 2-subset partitions of the beliefs
    partitions = fast_k_subset_partitions(list(range(m)), 2)
    ps = []

    for partition in partitions:
        m1 = partition[1]

        # get the aggregated opinion for this pole
        ys = beliefs[:, m1].sum(axis=1)

        # sort the aggregated opinions
        ys = ys.sort()[0]

        # find the highest polarization partition
        max_p = 0
        for i in range(1, n):
            yn0 = ys[:i]
            yn1 = ys[i:]

            # get the cross product difference of the two partitions
            p = (yn1.view(-1, 1).expand(-1, len(yn0)) - yn0.view(1, -1).expand(len(yn1), -1)).sum()
            if p > max_p:
                max_p = p
        
        ps.append(max_p / (4 * n**2))

    return ps

def fast_k_subset_partitions(ns, m):
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)