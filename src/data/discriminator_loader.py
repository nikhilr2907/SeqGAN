import numpy as np
import torch


class DisDataLoader:
    """Data loader for discriminator training."""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.num_batch = 0
        self.sentences_batches = []
        self.labels_batches = []
        self.pointer = 0

    def load_train_data(self, positive_file, negative_file):
        """Load positive and negative examples from files."""
        positive_examples = []
        negative_examples = []

        with open(positive_file) as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [float(x) for x in line]
                positive_examples.append(parse_line)

        with open(negative_file) as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [float(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        """Get next batch of sentences and labels."""
        ret = (
            torch.FloatTensor(self.sentences_batches[self.pointer]),
            torch.FloatTensor(self.labels_batches[self.pointer])
        )
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

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
        batch = (
            torch.FloatTensor(self.sentences_batches[self.pointer]),
            torch.FloatTensor(self.labels_batches[self.pointer])
        )
        self.pointer += 1
        return batch

    def __len__(self):
        """Return number of batches."""
        return self.num_batch
