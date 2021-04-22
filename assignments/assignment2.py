import numpy as np

from data import *
from utils import normalize, train_validation_error, translate
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


def best(reg_lambda):
    k = 3
    ns = 980

    classifier.reset()
    cost, loss, accuracy = classifier.fit(X_more_train, Y_more_train, X_more_val, Y_more_val, k=k, ns=ns,
                                          reg_lambda=reg_lambda)

    steps = 2 * ns * k
    train_validation_error(*cost, label='cost', steps=steps, save='best-reg-lambda')
    train_validation_error(*loss, label='loss', steps=steps, save='best-reg-lambda')
    train_validation_error(*accuracy, label='accuracy', steps=steps, save='best-reg-lambda')

    print(classifier.accuracy(X_more_test, Y_more_test))


def random_search():
    reg_lambda = np.random.uniform(-5, -1, 5)
    reg_lambda = 10 ** reg_lambda
    batch_size = [50, 100, 250, 500]
    ks = [2, 3, 5]

    for reg in reg_lambda:
        for bs in batch_size:
            for k in ks:
                classifier.reset()
                ns = 2 * np.floor(X_more_train.shape[1] / bs).astype(int)
                classifier.fit(X_more_train, Y_more_train, X_more_val, Y_more_val, k=k, ns=ns, reg_lambda=reg)

                print(f"Accuracy for {k} cycles, batch size {bs} and reg lambda {reg}:"
                      f" {classifier.accuracy(X_more_test, Y_more_test)}")


def hidden_nodes_vs_regularization():
    hidden_nodes = [50, 100, 200, 500, 1000]
    batch_size = 100
    ns = 2 * np.floor(X_more_train.shape[1] / batch_size).astype(int)
    reg_lambda = [10 ** l for l in np.linspace(-5, -1, 5)]

    for hn in hidden_nodes:
        c = Classifier()
        c.set_input(X_train.shape[0])
        c.add_layer(hn, "relu")
        c.add_layer(Y_train.shape[0], "softmax")
        for reg in reg_lambda:
            c.reset()
            c.fit(X_more_train, Y_more_train, X_more_val, Y_more_val, k=3, ns=ns, reg_lambda=reg)
            print(f"Accuracy for {hn} hidden nodes and reg lambda {reg}:"
                  f" {c.accuracy(X_more_test, Y_more_test)}")


def random_jitter():
    k = 3
    ns = 980

    c = Classifier()
    c.set_input(X_train.shape[0])
    c.add_layer(50, "relu")
    c.add_layer(Y_train.shape[0], "softmax")

    for jitter in [True, False]:
        c.reset()
        cost, loss, accuracy = c.fit(X_more_train, Y_more_train, X_more_val, Y_more_val, k=k, ns=ns, reg_lambda=0.001,
                                     jitter=jitter)

        steps = 2 * ns * k
        train_validation_error(*cost, label='cost', steps=steps, save=f'best-jitter-{jitter}')
        train_validation_error(*loss, label='loss', steps=steps, save=f'best-jitter-{jitter}')
        train_validation_error(*accuracy, label='accuracy', steps=steps, save=f'best-jitter-{jitter}')

        print(f"Accuracy with jitter {jitter}: {c.accuracy(X_more_test, Y_more_test)}")


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

    X_more_train, Y_more_train, _ = load_all_data()
    X_more_mean = np.mean(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_std = np.std(X_more_train, axis=1).reshape(X_more_train.shape[0], 1)
    X_more_train = normalize(X_more_train, X_more_mean, X_more_std)

    X_more_val, Y_more_val, _ = load_all_data(validation=True)
    X_more_test, Y_more_test, _ = load_data('test_batch')
    X_more_val = normalize(X_more_val, X_more_mean, X_more_std)
    X_more_test = normalize(X_more_test, X_more_mean, X_more_std)

    # lambda_search(-5, -1)
    # lambda_search(-6, -4.5)

    # best(3.162277660168379e-06)

    # random_search()

    # hidden_nodes_vs_regularization()

    random_jitter()
