from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def train_classifier(dataset, cv=3):
    X, y = [], []
    for text, label in dataset:
        assert isinstance(text, str)
        assert label in {True, False}
        X.append(text)
        y.append(label)
    print('Got {} labelled samples'.format(len(X)))
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('clf', SGDClassifier(loss='log')),
    ])
    if cv:
        print('Cross-validating')
        scores = cross_val_score(pipeline, X, y, cv=cv)
        print("Cross-validated accuracy: %0.2f (+/- %0.2f)" %
              (scores.mean(), scores.std() * 2))
    print('Refitting')
    pipeline.fit(X, y)
    return pipeline
