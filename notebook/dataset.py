from collections import Counter
import csv
import random


def pool_data(datasets):
    pooled = Dataset()
    for k, d in datasets:
        pooled.update(d)
    return pooled


class Dataset(object):
    """ Encapsulates unlabelled and labelled examples. """
    def __init__(self, text_to_label=None):
        self._oracle = text_to_label or {}
        self._annotation = {text: None for text, _
                           in self._oracle.items()}

    def add_label(self, text, label):
        self._annotation[text] = label

    def __len__(self):
        return len(self._annotation)

    def __iter__(self):
        return ((text, label) for text, label in
                self._annotation.items())

    def __getitem__(self, text):
        return self._annotation.get(text)

    def to_csv(self, fname):
        with open(fname, 'w') as f:
            w = csv.writer(f, delimiter=',')
            w.writerows(self.labelled_items)

    @classmethod
    def from_csv(cls, fname):
        with open(fname) as f:
            data = {}
            for row in csv.reader(f, delimiter=','):
                assert 1 <= len(row) <= 2
                if len(row) == 1:
                    text = row[0]
                    label = None
                else:
                    text, label = row
                    label = {
                        'True': True,
                        'False': False,
                        '': None,
                        'None': None,
                    }[label]
                data[text] = label
            return cls(data)

    def update(self, other):
        self._annotation.update(other._annotation)

    def get_oracle_label(self, text):
        return self._oracle[text]

    def seed(self, n):
        """ Seed with n labels from oracle. """
        unlabelled = list(self.unlabelled_items)
        random.shuffle(unlabelled)
        for i, (text, label) in enumerate(unlabelled):
            if i >= n:
                break
            label = self.get_oracle_label(text)
            self.add_label(text, label)

    @property
    def oracle_items(self):
        return self._oracle.items()

    @property
    def labelled_items(self):
        for text, label in self:
            if label in {True, False}:
                yield text, label

    @property
    def unlabelled_items(self):
        for text, label in self:
            if label is None:
                yield text, label

    @property
    def copy(self):
        d = Dataset()
        d._annotation = dict(self._annotation)  # copy
        d._oracle = dict(self._oracle)  # copy
        return d

    @property
    def label_distribution(self):
        return dict(Counter(self._annotation.values()))
