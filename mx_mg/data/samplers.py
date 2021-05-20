# Import packages
import numpy as np
import random
from mxnet.gluon.data.sampler import Sampler

__all__ = ['BalancedSampler']


class BalancedSampler(Sampler):
    def __init__(self, cost, batch_size):
        """
        Reorder the indexes of the samples according to a cost. In this case the smiles length. The batches returned
        are made up of molecules with different lengths.

        :param cost: list of the lengths of all the smiles string in the dataset.
        :param batch_size: number of samples per batch.
        :return:
        """
        # Get the indexes of the elements from the smallest length to the largest.
        index = np.argsort(cost).tolist()
        # Number of batches that divide the dataset
        chunk_size = int(float(len(cost)) / batch_size)
        self.index = []
        # Divide the dataset indexes in batch size separate lists. The first list comprises molecules with smaller
        # lengths, the latter the opposite.
        for i in range(batch_size):
            self.index.append(index[i * chunk_size:(i + 1) * chunk_size])

    def _g(self):
        # Shuffle every list of indexes
        for index_i in self.index:
            random.shuffle(index_i)

        # Gather the elements of every list that are in the same position
        for batch_index in zip(*self.index):
            # A tuple of batch_size indexes is returned
            yield batch_index

    def __iter__(self):
        return self._g()

    def __len__(self):
        return len(self.index[0])
