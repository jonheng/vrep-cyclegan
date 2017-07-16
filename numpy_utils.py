import numpy as np


def random_indices(size):
    x = np.arange(size)
    np.random.shuffle(x)
    return x


class RandomBatchIndices:
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.total_steps = np.ceil(float(dataset_size) / batch_size)
        self.sequence = random_indices(dataset_size)
        self.counter = 0

    def next_batch_index(self):
        if self.counter == self.total_steps:
            return None
        next_indices = self.sequence[self.counter * self.batch_size : (self.counter + 1) * self.batch_size]
        self.counter += 1
        return next_indices



if __name__=="__main__":
    print("Numpy utils main")
    print("Shuffling np.arange(10)")
    print random_indices(10)
