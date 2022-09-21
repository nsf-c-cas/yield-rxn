import random
import torch

from torch.utils.data import Sampler

class BinRandomSampler(Sampler):
    """Samples elements randomly from a dictionary of bin sizes and lists
    of indices corresponding to the master list of reactions for the dataset.

    Arguments:
        indices (dict): a sequence of indices
    """

    def __init__(self, indices, batch_size):
        super(BinRandomSampler, self).__init__()
        self.indices = indices
        self.batch_size = batch_size

    def __iter__(self):
        # Get the random permutations of the binned indices.
        rand_bins = [torch.randperm(len(idx_bin)) for idx_bin in self.indices.values()]
        # Trim the permuted indices so that each set is divisible by the batch size.
        rand_bins = [idx_bin[:-1 * (len(idx_bin) % self.batch_size)] for idx_bin in rand_bins]
        # Now collect batch size chunks of indices from each bin into a master list.
        batches = []
        for idx_bin in rand_bins:
            for i in range(len(idx_bin) / self.batch_size):
                batches.append([idx_bin[i*self.batch_size:(i+1)*self.batch_size]])
        # Shuffle to keep batches together but order of batches randomized.
        random.shuffle(batches)
        # Then merge into one list of indices for the current epoch.
        return [idx for batch in batches for idx in batch]


        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return sum([len(bin) for bin in self.indices.values()])
