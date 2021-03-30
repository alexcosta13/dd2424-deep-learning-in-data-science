import numpy as np

from classifier import Classifier
from data import *
from utils import normalize, one_hot_encode_labels, montage, train_validation_error, show_image


BATCH_SIZE = 20


def load_data(file_name):
    train_data = load_batch(file_name)
    X, Y = dict_to_data_and_label(train_data)
    Y = one_hot_encode_labels(Y)
    y = load_label_names()
    return X.T, Y.T, y


def check_gradients(c, data, reg=0, slow=False):
    grad_W, grad_b = c.compute_gradients(*data, reg)
    if not slow:
        grad_W_num, grad_b_num = c.compute_gradients_num(*data, reg)
    else:
        grad_W_num, grad_b_num = c.compute_gradients_num_slow(*data, reg)  # no funciona
    equal = np.allclose(grad_W, grad_W_num, rtol=1e-6, atol=1e-6) and np.allclose(grad_b, grad_b_num, rtol=1e-6, atol=1e-6)
    threshold = max(np.max(np.abs(grad_W - grad_W_num)), np.max(np.abs(grad_b - grad_b_num)))
    return equal, threshold


def explore_images(n, input_data):
    for i in range(n):
        show_image(input_data[:, i])


if __name__ == "__main__":
    # explore_images(5, X_train)

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
    print('Cost', classifier.compute_cost(*batch, 0))
    # print('Numerical gradients', check_gradients(classifier, batch))
    # test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val)
    # train_validation_error(test_error, val_error)
    montage(classifier.W)
    print('Classifier accuracy', classifier.accuracy(X_test, Y_test)*100, '%')

    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.1, reg_lambda=0)
    train_validation_error(test_error, val_error)
    montage(classifier.W)
    print('Classifier accuracy', classifier.accuracy(X_test, Y_test) * 100, '%')
    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.001, reg_lambda=0)
    train_validation_error(test_error, val_error)
    montage(classifier.W)
    print('Classifier accuracy', classifier.accuracy(X_test, Y_test) * 100, '%')
    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.001, reg_lambda=0.1)
    train_validation_error(test_error, val_error)
    montage(classifier.W)
    print('Classifier accuracy', classifier.accuracy(X_test, Y_test) * 100, '%')
    test_error, val_error = classifier.fit(X_train, Y_train, X_val, Y_val, eta=0.001, reg_lambda=1)
    train_validation_error(test_error, val_error)
    montage(classifier.W)
    print('Classifier accuracy', classifier.accuracy(X_test, Y_test) * 100, '%')
