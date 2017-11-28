import random


random.seed(1)


class Random(object):
    def __init__(self, limit=None):
        self.limit = limit

    def __call__(self, dataset):
        unlabelled = list(((text, label)
                          for (text, label) in dataset
                          if label is None))
        random.shuffle(unlabelled)
        yielded = 0
        for text, label in unlabelled:
            yield text, label
            yielded += 1
            if self.limit and yielded == self.limit:
                break


class Active(object):
    def __init__(self, pipeline,
                 query=lambda i: i[1] is None,
                 accept=lambda i: True, limit=None):
        self.pipeline = pipeline
        self.limit = limit
        self.query = query
        self.accept = accept

    def __call__(self, dataset):
        X, y = [], []
        for text, label in dataset:
            if self.query((text, label)):
                X.append(text)
                y.append(label)
        print('Predicting {} unlabelled'.format(len(X)))
        yielded = 0
        for probs, text in zip(self.pipeline.predict_proba(X), X):
            predictions = dict(zip(self.pipeline.classes_, probs))
            if self.accept(predictions):
                yield text, predictions
                yielded += 1
                if self.limit and yielded == self.limit:
                    break
