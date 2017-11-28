from collections import Counter
import csv
import random

from sklearn.datasets import fetch_20newsgroups


class Dataset(object):
    """ Encapsulates unlabelled and labelled examples. """
    def __init__(self, text_to_label=None):
        self.text_to_label = text_to_label or {}

    def add_label(self, text, label):
        self.text_to_label[text] = label

    def __iter__(self):
        return ((text, label) for text, label in
                self.text_to_label.items())

    def to_csv(self, fname):
        with open(fname, 'w') as f:
            w = csv.writer(f, delimiter=',')
            for text, label in self.text_to_label.items():
                w.writerow((label, text))

    @classmethod
    def from_csv(cls, fname):
        with open(fname) as f:
            return cls({text: label for text, label in
                        csv.reader(f, delimiter=',')})

    def update(self, other):
        self.text_to_label.update(other.text_to_label)

    @property
    def label_distribution(self):
        return dict(Counter(self.text_to_label.values()))


def build_sample_corpus(subset='train'):
    """ Fetches newsgroupd data and returns a Dataset. """
    newsgroups_train = fetch_20newsgroups(subset=subset)
    label_names = {index: name for index, name in
                   enumerate(newsgroups_train.target_names)}
    # Transform to guns or not.
    for i, name in list(label_names.items()):
        label_names[i] = name == 'talk.politics.guns'
    dataset = Dataset({text: label_names[index]
                      for text, index in zip(newsgroups_train.data,
                                             newsgroups_train.target)})
    unlabel(dataset)
    return dataset


def unlabel(dataset, p=0.01):
    """ Randomly removes some labels. """
    for text, label in dataset:
        if random.random() > p:
            dataset.add_label(text, None)
