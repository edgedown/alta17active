from sklearn.metrics import f1_score, make_scorer
import itertools
import matplotlib.pyplot as plt
import numpy as np
import random

from dataset import Dataset


f1_scorer = make_scorer(f1_score)


def plot_learning_curve(train_sizes, train_scores, test_scores):
    plt.clf()
    plt.figure()
    plt.xlabel("N training examples")
    plt.ylabel("F1 score")
    plt.ylim((0.0, 1.0))
    train_sizes_mean = np.mean(train_sizes, axis=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    
    plt.fill_between(train_sizes_mean, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes_mean, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes_mean, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes_mean, test_scores_mean, 'o-', color="g",
             label="Test score")
    
    plt.legend(loc="best")
    return plt


def run_simulations(sampler, pool, test, n=5):
    """
    Run n simulated learning-curve experiments.
    
    sampler - an function for sampling from pool
    pool - pool dataset (with oracle labels)
    test - test dataset (with oracle labels)
    n - number of bootstraps for confidence intervals (default=10)
    seed_size - seed pool with this many labelled items
    """
    # run simulations
    runs = []
    for i in range(n):
        print('Running simulation {}..'.format(i))
        runs.append(zip(*list(_run_simulation(sampler, pool.copy, test))))
    # return train_sizes, train_scores, test_scores
    return (list(zip(*i)) for i in zip(*runs))


def _run_simulation(sampler, pool, test, seed_size=100):
    """ Yield train_size, train_score, test_score per batch. """
    # get test data
    X_test, y_test = zip(*test.oracle_items)
    # seed with batch_size labels before training
    pool.seed(sampler.batch_size)
    # sample until pool is empty, yielding train/test f1
    for i in itertools.count():
        print('..batch {}..'.format(i))
        # evaluate
        yield next(sampler.fit_and_score(pool, X_test, y_test))
        # get next batch
        batch = list(sampler(pool))
        # stop if no more data to sample
        if not batch:
            break
        # add batch examples with oracle labels
        for text, label in sampler(pool):
            label = pool.get_oracle_label(text)
            pool.add_label(text, label)


def run_bootstraps(sampler, pool, test, n=3):
    # get test data
    X_test, y_test = zip(*test.oracle_items)
    # seed and evaluate
    pool.seed(sampler.batch_size)
    # sample and bootstrap
    batches = []
    for i in itertools.count():
        print('Running batch {}..'.format(i))
        # evaluate with n bootstrap resamples of labelled data
        batches.append(list(sampler.fit_and_score(pool, X_test, y_test, n)))
        # get next batch
        batch = list(sampler(pool))
        # stop if no more data to sample
        if not batch:
            break
        # add batch examples with oracle labels
        for text, label in batch:
            label = pool.get_oracle_label(text)
            pool.add_label(text, label)
    return zip(*[zip(*i) for i in batches])


def fit_and_score(pipeline, pool, X_test, y_test, n=1):
    """                                                                                                              
    Fit model to labelled data from pool and score on train and test.                                                
    If n>1, then run n times with bootstrapped training data.                                                        
    """
    for i in range(n):
        X_train, y_train = zip(*pool.labelled_items)
        if n > 1:
            X_boot, y_boot = zip(*resample(X_train, y_train))
        else:
            X_boot, y_boot = X_train, y_train
        print('..fitting to {} labelled examples..'.format(len(X_train)))
        pipeline.fit(X_boot, y_boot)
        f1_boot = f1_scorer(pipeline, X_boot, y_boot)
        f1_test = f1_scorer(pipeline, X_test, y_test)
        yield len(X_boot), f1_boot, f1_test


def resample(X, y):
    """ Yield len(X) items sampled with replacement from X, y. """
    N = len(X)
    for _ in range(N):
        i = random.randint(0, N - 1)
        yield X[i], y[i]


def label_for_submission(dataset, clf, dataset_name, username='username'):
    assert isinstance(dataset, Dataset)
    assert hasattr(clf, 'predict')
    assert dataset_name in {'dev', 'test'}
    dataset_preds = dataset.copy
    to_predict = list(i[0] for i in dataset_preds.unlabelled_items)
    labels = clf.predict(to_predict)
    predictions = dict(zip(to_predict, labels))
    for text, label in dataset_preds.unlabelled_items:
        dataset_preds.add_label(text, predictions[text])
    fname = '../submissions/{}/{}.csv'.format(username, dataset_name)
    dataset_preds.to_csv(fname)
    print('Written submission to {}'.format(fname))
