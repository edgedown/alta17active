from classifier import train_classifier
from evaluation import f1_scorer
import random


class Sampler(object):
    def __init__(self, batch_size=30,
                 query=lambda i: i[1] is None, accept=lambda i: True):
        self.batch_size = batch_size  # number of items per batch
        self.query = query  # a function to filter incoming data
        self.accept = accept  # a function to filter predictions
        self.pipeline = None  # a scikit-learn classifier

    def __call__(self, dataset):
        """ Sample a batch from dataset. """
        raise NotImplementedError

    def fit(self, pool):
        X_train, y_train = zip(*pol.labelled_items)
        self._fit(X_train, y_train)

    def _fit(self, X, y):
        self.pipeline = train_classifier(X, y, cv=0)

    def f1_score(self, X, y):
        return f1_scorer(self.pipeline, X, y)

    def fit_and_score(self, pool, X_test=None, y_test=None):
        X_train, y_train = zip(*pool.labelled_items)
        self._fit(X_train, y_train)
        f1_train = self.f1_score(X_train, y_train)
        f1_test = self.f1_score(X_test, y_test) if X_test and y_test else None
        return len(X_train), f1_train, f1_test


class Random(Sampler):
    def __call__(self, dataset):
        unlabelled = list(dataset.unlabelled_items)
        random.shuffle(unlabelled)
        for i, (text, label) in enumerate(unlabelled):
            if i >= self.batch_size:
                break
            yield text, label


class Active(Sampler):
    def __call__(self, dataset):
        X, y = [], []
        for text, label in dataset:
            if self.query((text, label)):
                X.append(text)
                y.append(label)
        print('Predicting {} unlabelled'.format(len(X)))
        count = 0
        for probs, text in zip(self.pipeline.predict_proba(X), X):
            predictions = dict(zip(self.pipeline.classes_, probs))
            if self.accept(predictions):
                yield text, predictions
                count += 1
                if self.batch_size and count == self.batch_size:
                    break
