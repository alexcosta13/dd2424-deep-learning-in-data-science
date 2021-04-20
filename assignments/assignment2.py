import numpy as np

from data import *
from utils import normalize, train_validation_error
from two_layer_classifier import Classifier
from assignment1 import load_all_data

HIDDEN_NODES = 50
BATCH_SIZE = 20


def check_gradients(reg=0.0, slow=False):
    grad_W2, grad_b2, grad_W1, grad_b1 = classifier.compute_gradients(*batch, reg)
    if not slow:
        grad_W2_num, grad_b2_num, grad_W1_num, grad_b1_num = classifier.compute_gradients_num(*batch, reg)
    else:
        grad_W2_num, grad_b2_num, grad_W1_num, grad_b1_num = classifier.compute_gradients_num_slow(*batch, reg)
    equal = np.allclose(grad_W2, grad_W2_num, rtol=1e-6, atol=1e-6) and np.allclose(grad_b2, grad_b2_num, rtol=1e-6,
                                                                                    atol=1e-6)
    equal = equal and np.allclose(grad_W1, grad_W1_num, rtol=1e-6, atol=1e-6) and np.allclose(grad_b1, grad_b1_num,
                                                                                              rtol=1e-6, atol=1e-6)
    threshold = max(np.max(np.abs(grad_W2 - grad_W2_num)), np.max(np.abs(grad_b2 - grad_b2_num)),
                    np.max(np.abs(grad_W1 - grad_W1_num)), np.max(np.abs(grad_b1 - grad_b1_num)))
    return equal, threshold


def basic_assignment(k, ns=500):
    classifier.reset()
    cost, loss, accuracy = classifier.fit(X_train, Y_train, X_val, Y_val, k=k, ns=ns, reg_lambda=0.01)

    steps = 2 * ns * k
    train_validation_error(*cost, label='cost', steps=steps, save=f'{k}-cycles')
    train_validation_error(*loss, label='loss', steps=steps, save=f'{k}-cycles')
    train_validation_error(*accuracy, label='accuracy', steps=steps, save=f'{k}-cycles')

    return classifier.accuracy(X_test, Y_test)


def lambda_search(l_min, l_max, random=False):
    if random:
        reg_lambda = np.random.uniform(l_min, l_max, 5)
        reg_lambda = 10 ** reg_lambda
    else:
        reg_lambda = [10 ** l for l in np.linspace(l_min, l_max, 10)]

    for i in reg_lambda:
        classifier.reset()
        classifier.fit(X_more_train, Y_more_train, k=3, ns=980, reg_lambda=i)
        print(f"Accuracy for regularization lambda={i}: {classifier.accuracy(X_more_test, Y_more_test)*100}%")


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

    batch = X_train[:, :BATCH_SIZE], Y_train[:, :BATCH_SIZE]

    classifier = Classifier()
    classifier.set_input(X_train.shape[0])
    classifier.add_layer(HIDDEN_NODES, "relu")
    classifier.add_layer(Y_train.shape[0], "softmax")

    # print('Numerical gradients', check_gradients())
    # print('Slow numerical gradients', check_gradients(slow=True))

    # print('Accuracy after 1 cycle and ns=500', basic_assignment(1, 500))
    # print('Accuracy after 3 cycles and ns=800', basic_assignment(3, 800))

    X_more_train, Y_more_train, y = load_all_data()
    X_more_mean = np.mean(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_std = np.std(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_train = normalize(X_more_train, X_more_mean, X_more_std)

    X_more_val, Y_more_val, _ = load_all_data(validation=True)
    X_more_test, Y_more_test, _ = load_data('test_batch')
    X_more_val = normalize(X_more_val, X_more_mean, X_more_std)
    X_more_test = normalize(X_more_test, X_more_mean, X_more_std)

    # lambda_search(-5, -1)
    # lambda_search(-6, -4.5)

    k = 3
    ns = 980

    classifier.reset()
    cost, loss, accuracy = classifier.fit(X_more_train, Y_more_train, X_more_val, Y_more_val, k=k, ns=ns,
                                          reg_lambda=3.162277660168379e-06)

    steps = 2 * ns * k
    train_validation_error(*cost, label='cost', steps=steps, save='best-reg-lambda')
    train_validation_error(*loss, label='loss', steps=steps, save='best-reg-lambda')
    train_validation_error(*accuracy, label='accuracy', steps=steps, save='best-reg-lambda')

    print(classifier.accuracy(X_more_test, Y_more_test))
