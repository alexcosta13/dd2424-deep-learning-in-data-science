import numpy as np

from classifier import Classifier
from data import *
from utils import normalize, montage, train_validation_error


BATCH_SIZE = 20


def check_gradients(reg=0.0, slow=False):
    grad_W, grad_b = classifier.compute_gradients(*batch, reg)
    if not slow:
        grad_W_num, grad_b_num = classifier.compute_gradients_num(*batch, reg)
    else:
        grad_W_num, grad_b_num = classifier.compute_gradients_num_slow(*batch, reg)  # no funciona
    equal = np.allclose(grad_W, grad_W_num, rtol=1e-6, atol=1e-6) and np.allclose(grad_b, grad_b_num, rtol=1e-6, atol=1e-6)
    threshold = max(np.max(np.abs(grad_W - grad_W_num)), np.max(np.abs(grad_b - grad_b_num)))
    return equal, threshold


def basic_assignment():
    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.1, reg_lambda=0)
    train_validation_error(test_error, val_error, title=r'Training and validation error for $\eta=0.1$ and $\lambda=0$',
                           save="eta01-lambda0")
    montage(classifier.W, y, title=r'Learnt weights for $\eta=0.1$ and $\lambda=0$', save="eta01-lambda0")
    print("Classifier accuracy (eta=0.1 and lambda=0)", "{:.2%}".format(classifier.accuracy(X_test, Y_test)))

    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.001, reg_lambda=0)
    train_validation_error(test_error, val_error, title=r'Training and validation error for $\eta=0.001$ and '
                                                        r'$\lambda=0$', save="eta0001-lambda0")
    montage(classifier.W, y, title=r'Learnt weights for $\eta=0.001$ and $\lambda=0$', save="eta0001-lambda0")
    print("Classifier accuracy (eta=0.001 and lambda=0)", "{:.2%}".format(classifier.accuracy(X_test, Y_test)))

    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.001, reg_lambda=0.1)
    train_validation_error(test_error, val_error, title=r'Training and validation error for $\eta=0.001$ and '
                                                        r'$\lambda=0.1$', save="eta0001-lambda01")
    montage(classifier.W, y, title=r'Learnt weights for $\eta=0.001$ and $\lambda=0.1$', save="eta0001-lambda01")
    print("Classifier accuracy (eta=0.001 and lambda=0.1)", "{:.2%}".format(classifier.accuracy(X_test, Y_test)))

    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.001, reg_lambda=1)
    train_validation_error(test_error, val_error, title=r'Training and validation error for $\eta=0.001$ and '
                                                        r'$\lambda=1$', save="eta0001-lambda1")
    montage(classifier.W, y, title=r'Learnt weights for $\eta=0.001$ and $\lambda=1$', save="eta0001-lambda1")
    print("Classifier accuracy (eta=0.001 and lambda=1)", "{:.2%}".format(classifier.accuracy(X_test, Y_test)))


def more_training_data():
    X_more_train, Y_more_train, y = load_all_data()
    X_more_mean = np.mean(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_std = np.std(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_train = normalize(X_more_train, X_more_mean, X_more_std)

    X_more_val, Y_more_val, _ = load_all_data(validation=True)
    X_more_test, Y_more_test, _ = load_data('test_batch')
    X_more_val = normalize(X_more_val, X_more_mean, X_more_std)
    X_more_test = normalize(X_more_test, X_more_mean, X_more_std)

    test_error, val_error = classifier.fit(X_more_train, Y_more_train, X_more_val, Y_more_val, eta=0.001, reg_lambda=0.1)
    train_validation_error(test_error, val_error, title=r'Training and validation error for $\eta=0.001$ and '
                                                        r'$\lambda=0.1$', save="eta0001-lambda01-extra-data")
    montage(classifier.W, y, title=r'Learned weights for $\eta=0.001$ and $\lambda=0.1$', save="eta0001-lambda01-extra-data")
    print("Classifier accuracy (eta=0.001 and lambda=0.1, more training data)", "{:.2%}".format(classifier.accuracy(X_more_test, Y_more_test)))


def grid_search():
    regularization = [0.1, 0.25, 0.5, 1, 1.5]
    learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    batch_size = [25, 50, 100, 500, 1000]

    acc = {}

    for reg_lambda in regularization:
        for rate in learning_rate:
            for size in batch_size:
                classifier.fit(X_train, Y_train, X_val, Y_val, n_batch=size, eta=rate, reg_lambda=reg_lambda)
                acc[f'learning_rate: {rate} batch_size: {size} reg_lambda: {reg_lambda}'] = classifier.accuracy(X_test, Y_test)
                print(reg_lambda, rate, size)

    best = max(acc, key=acc.get)
    print(f'{best}, accuracy: {acc[best]}')


def shuffle():
    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.001, reg_lambda=0.1, shuffle=True,
                                           n_epochs=100)
    train_validation_error(test_error, val_error, title=r'Training and validation error for $\eta=0.1$ and $\lambda=0$',
                           save="eta0001-lambda01-shuffle")
    montage(classifier.W, y, title=r'Learned weights for $\eta=0.1$ and $\lambda=0$', save="eta0001-lambda01-shuffle")
    print("Classifier accuracy (eta=0.001 and lambda=0.1, shuffling)", "{:.2%}".format(classifier.accuracy(X_test, Y_test)))


def decay_learning_rate():
    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.1, reg_lambda=0.1, shuffle=True,
                                           n_epochs=100, decay_factor=0.9)
    train_validation_error(test_error, val_error, title=r'Training and validation error for $\eta=0.1$ and $\lambda=0$',
                           save="eta0001-lambda01-decay")
    montage(classifier.W, y, title=r'Learned weights for $\eta=0.1$ and $\lambda=0$', save="eta0001-lambda01-decay")
    print("Classifier accuracy (eta=0.001 and lambda=0.1, decaying)", "{:.2%}".format(classifier.accuracy(X_test, Y_test)))


def all_optimizations():
    X_more_train, Y_more_train, y = load_all_data()
    X_more_mean = np.mean(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_std = np.std(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_train = normalize(X_more_train, X_more_mean, X_more_std)

    X_more_val, Y_more_val, _ = load_all_data(validation=True)
    X_more_test, Y_more_test, _ = load_data('test_batch')
    X_more_val = normalize(X_more_val, X_more_mean, X_more_std)
    X_more_test = normalize(X_more_test, X_more_mean, X_more_std)

    test_error, val_error = classifier.fit(X_more_train, Y_more_train, X_more_val, Y_more_val, eta=0.01, reg_lambda=0.1,
                                           shuffle=True, decay_factor=0.9)
    train_validation_error(test_error, val_error, title=r'Training and validation error with all previous optimizations',
                           save="optimizations-all")
    montage(classifier.W, y, title=r'Learned weights with all previous optimizations', save="optimizations-all")
    print("Classifier accuracy (eta=0.001 and lambda=0.1, more training data)", "{:.2%}".format(classifier.accuracy(X_more_test, Y_more_test)))


def svm():
    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, loss_function='svm', n_epochs=50, n_batch=1000,
                                           eta=0.001)

    train_validation_error(test_error, val_error, title=r'Training and validation error', save='svm')
    montage(classifier.W, y, title=r'Learned weights', save="svm")
    print("Classifier accuracy (eta=0.001 and lambda=0.1)", "{:.2%}".format(classifier.accuracy(X_test, Y_test)))


if __name__ == "__main__":
    X_train, Y_train, y = load_data('data_batch_1')
    X_mean = np.mean(X_train, axis=1).reshape(X_train.shape[0], 1)
    X_std = np.std(X_train, axis=1).reshape(X_train.shape[0], 1)
    X_train = normalize(X_train, X_mean, X_std)

    assert np.allclose(np.mean(X_train, axis=1).reshape(X_train.shape[0], 1), np.zeros((X_train.shape[0], 1))), \
        "Check normalization, mean should be 0 "
    assert np.allclose(np.std(X_train, axis=1).reshape(X_train.shape[0], 1), np.ones((X_train.shape[0], 1))), \
        "Check normalization, std should be 1"

    X_val, Y_val, _ = load_data('data_batch_2')
    X_test, Y_test, _ = load_data('test_batch')
    X_val = normalize(X_val, X_mean, X_std)
    X_test = normalize(X_test, X_mean, X_std)

    classifier = Classifier(X_train.shape[0], Y_train.shape[0])
    batch = X_train[:, :BATCH_SIZE], Y_train[:, :BATCH_SIZE]

    # explore_images(5, X_train)

    # print('Numerical gradients', check_gradients(0.1))
    # print('Slow numerical gradients', check_gradients(slow=True))

    # basic_assignment()

    # more_training_data()

    # grid_search()

    # shuffle()

    # decay_learning_rate()

    # all_optimizations()

    svm()
