import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x ** 2


class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.__activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.__activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self._weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            # HIDDEN LAYER
            self._weights.append(r)
            print(self._weights)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        # OUTPUT LAYER
        self._weights.append(r)
        print(self._weights)

    def calculate_correct_classified_rate(self, predicted_y, real_y):
        correct_classified_number = 0
        for (y_guess, y_real) in zip(predicted_y, real_y):
            if y_real == 1:
                if y_guess > 0.5:
                    correct_classified_number += 1
            elif y_real == 0:
                if y_guess < 0.5:
                    correct_classified_number += 1

        correct_classifitation_rate = correct_classified_number / float(len(real_y))
        return correct_classifitation_rate

    def __calculate_test_error(self, X_test, real_y_test, X_train, real_y_train):

        predicted_y_test = [self.predict(e).ravel() for e in X_test]
        correct_classifitation_rate_on_test_set = self.calculate_correct_classified_rate(real_y=real_y_test, predicted_y=predicted_y_test)
        #predicted_y_train =  [self.predict(e).ravel() for e in X_train[:,1:]]
        #correct_classifitation_rate_on_train_set = self.calculate_correct_classified_rate(real_y=real_y_train, predicted_y=predicted_y_train)
        #print(correct_classifitation_rate_on_test_set, correct_classifitation_rate_on_train_set)

        return correct_classifitation_rate_on_test_set

    def fit(self, X, y, X_test, y_test, learning_rate=0.05, epochs=50000000000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            if k % 10000 == 1:
                print ('epochs:', k)
                if self.__calculate_test_error(X_test=X_test, real_y_test=y_test, X_train=X, real_y_train=y)/k <= 0.00000001:
                    break

            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self._weights)):
                dot_value = np.dot(a[l], self._weights[l])
                activation = self.__activation(dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self._weights[l].T) * self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self._weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self._weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=0)
        for l in range(0, len(self._weights)):
            a = self.__activation(np.dot(a, self._weights[l]))
        return a

    def get_weights(self):
        return self._weights


if __name__ == '__main__':

    nn = NeuralNetwork([4, 2, 1])

    X = np.array([[0, 0, 0.5, 0.5],
                  [0, 1, 0.5, 0.5],
                  [1, 0, 0.5, 0.5],
                  [1, 1, 0.5, 0.5]])

    y = np.array([0, 1, 1, 0])

    # nn.fit(X, y)
    #
    # X_test = np.array([[0, 0, 0.4, 0.7],
    #               [0, 1, 0.3, 0.8],
    #               [1, 0, 1, 0],
    #               [1, 1, 0.2, 0.6]])
    #
    # for e in X_test:
    #     print(e, nn.predict(e))