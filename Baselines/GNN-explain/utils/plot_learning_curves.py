import matplotlib.pyplot as plt

import numpy as np



def plot_learning_curve(samples, title="", axes=None, i=None, j=None, ylim=None, indexes=[0]):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1,2, figsize=(6,3))

    #axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    #axes[0].set_xlabel("Training examples")
    #axes[0].set_ylabel("Score")

    train_scores = np.array([[el[0][0] for el in s] for s in samples]).transpose()
    test_scores = np.array([[el[1][0] for el in s] for s in samples]).transpose()
    train_sizes = list(range(len(train_scores)))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    axes[i, j].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
    axes[i, j].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
    axes[i, j].plot(train_sizes, train_scores_mean, 'o-', color="r")
    # , label="Training loss")
    axes[i, j].plot(train_sizes, test_scores_mean, 'o-', color="g")
    # ,label="test-score loss")
    # axes[i,2*j].legend(loc="best")
    values = [(train_scores_mean, train_scores_std), (test_scores_mean, test_scores_std)]

    for i in indexes:
        train_lookup = np.array([[el[0][i] for el in s] for s in samples]).transpose()
        test_lookup = np.array([[el[1][i] for el in s] for s in samples]).transpose()

        train_lookup_mean = np.mean(train_lookup, axis=1)
        train_lookup_std = np.std(train_lookup, axis=1)
        test_lookup_mean = np.mean(test_lookup, axis=1)
        test_lookup_std = np.std(test_lookup, axis=1)
        #fit_times_mean = np.mean(fit_times, axis=1)
        #fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        #axes[i,2*j].grid()
        values += [(train_lookup_mean, train_lookup_std), (test_lookup_mean,test_lookup_std )]
        """
        "#axes[i,2*j+1].grid()
        axes[i, 2*j+1].fill_between(train_sizes, train_lookup_mean - train_lookup_std,
                             train_lookup_mean + train_lookup_std, alpha=0.1,
                             color="r")
        axes[i, 2*j+1].fill_between(train_sizes, test_lookup_mean - test_lookup_std,
                             test_lookup_mean + test_lookup_std, alpha=0.1,
                             color="g")
        axes[i, 2*j+1].plot(train_sizes, train_lookup_mean, 'o-', color="r")
        #,label="Training score")
        axes[i, 2*j+1].plot(train_sizes, test_lookup_mean, 'o-', color="g")
        #,label="test-score score")
        #axes[i, 2*j+1].legend(loc="best")
        """


    # Plot n_samples vs fit_times

    #plt.show()
    return values
