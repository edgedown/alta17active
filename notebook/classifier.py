from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


def train_classifier(X, y, cv=3):
    assert all(isinstance(text, str) for text in X)
    assert all(label in {True, False} for label in y)
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
