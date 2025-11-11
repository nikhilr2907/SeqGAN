import numpy as np
import torch


class GenDataLoader:
    """Data loader for generator training."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []
        self.num_batch = 0
        self.sequence_batch = []
        self.pointer = 0

    def create_batches(self, data_file):
        """Load data from file and create batches."""
        self.token_stream = []
        with open(data_file, "r") as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [float(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        """Get next batch of data."""
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return torch.FloatTensor(ret)

    def reset_pointer(self):
        """Reset batch pointer to beginning."""
        self.pointer = 0

    def __iter__(self):
        """Make the loader iterable."""
        self.reset_pointer()
        return self

    def __next__(self):
        """Get next batch for iteration."""
        if self.pointer >= self.num_batch:
            raise StopIteration
        batch = self.sequence_batch[self.pointer]
        self.pointer += 1
        return torch.FloatTensor(batch)

    def __len__(self):
        """Return number of batches."""
        return self.num_batch
