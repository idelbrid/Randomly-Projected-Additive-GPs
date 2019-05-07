#
class MCMCSampler(object):
    def __init__(self):
        self.burn_in = None
        pass

    def sample(self, n_samples=1):
        pass

    def _sample(self):
        pass

    def predict(self, x):
        pass

class HMSampler(MCMCSampler):
    def __init__(self):
        super(HMSampler, self).__init__()

    def _propose(self):
        pass

    def _accept(self):
        pass

    def _sample(self):
        self._propose()

class GardnerMHSampler(HMSampler):
    def __init__(self):
        super(GardnerMHSampler, self).__init__()

    def _propose(self, ):
        pass
            # split = np.random.rand() < 0.5
            # if split:
            #     notsplit = True
            #     while notsplit:
            #         idx = np.random.randint(0, len(degrees))
            #         if degrees[idx] > 1:
            #             notsplit = False
            #             proposed = []
            #             for i, deg in enumerate(degrees):
            #                 if i == idx:
            #                     first = np.random.randint(1, deg)
            #                     proposed.append(first)
            #                     proposed.append(deg - first)
            #                 else:
            #                     proposed.append(deg)
            #     prob_forward = 0.5 * (1 / len(degrees)) * 2 ** (-degrees[idx] + 1)  # adjust!!!
            #     prob_backward = 0.5 * (1 / len(proposed)) * (1 / (len(proposed) - 1))
            # else:
            #     idx1, idx2 = np.random.choice(np.arange(len(degrees)), size=2, replace=False)
            #     proposed = [degrees[idx1] + degrees[idx2]]
            #     for i, deg in enumerate(degrees):
            #         if i != idx1 and i != idx2:
            #             proposed.append(deg)
            #
            #     prob_forward = 0.5 * (1 / len(degrees)) * (1 / (len(degrees) - 1))
            #     prob_backward = 0.5 * (1 / len(proposed)) * 2 ** (-proposed[0] + 1)  # adjust?
            # if split:
            #     print("SPLIT {} TO {}".format(degrees, proposed))
            # else:
            #     print("MERGE {} TO {}".format(degrees, proposed))
            # return proposed, prob_forward, prob_backward



