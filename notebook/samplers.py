from evaluation import fit_and_score
import random


def _query(item):
    """ Default query. """
    return True


def _accept(pred):
    """ Default acceptor. """
    return True


class Sampler(object):
    def __init__(self, pipeline, batch_size=30, query=_query,
                 key=None, accept=_accept):
        self.pipeline = pipeline  # a sklearn pipeline or classifier
        self.batch_size = batch_size  # number of items per batch
        self.query = query  # a function to filter before prediction
        self.key = key  # a key function to sort predictions
        self.accept = accept  # a function to filter predictions

    def __call__(self, dataset):
        """ Sample a batch from dataset. """
        raise NotImplementedError

    def fit(self, pool):
        X_train, y_train = zip(*pool.labelled_items)
        self.pipeline.fit(X_train, y_train)

    def fit_and_score(self, pool, X_test, y_test, n=1):
        """
        Fit model to labelled data from pool and score on train and test.
        If n>1, then run n times with bootstrapped training data.
        """
        return iter(fit_and_score(self.pipeline, pool, X_test, y_test, n=n))


class Random(Sampler):
    def __call__(self, dataset):
        unlabelled = list(filter(self.query, dataset.unlabelled_items))
        random.shuffle(unlabelled)
        for i, (text, label) in enumerate(unlabelled):
            if i >= self.batch_size:
                break
            yield text, label


class Active(Sampler):
    def __call__(self, dataset):
        unlabelled = list(filter(self.query, dataset.unlabelled_items))
        if unlabelled:
            X, _ = zip(*unlabelled)
            count = 0
            predictions = zip(self.pipeline.predict_proba(X), X)
            if self.key:
                predictions = sorted(predictions, key=self.key)
            for probs, text in predictions:
                pred = dict(zip(self.pipeline.classes_, probs))
                if self.accept(pred):
                    yield text, pred
                    count += 1
                    if self.batch_size and count == self.batch_size:
                        break
