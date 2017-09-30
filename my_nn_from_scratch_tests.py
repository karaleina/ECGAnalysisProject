# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# https://stackoverflow.com/questions/34829807/understand-how-this-lambda-function-works

from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from neural_model_functions.simple_neural_models import plot_decision_boundary, build_model, predict


# Creating dataset
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# Linear regression
clf = linear_model.LogisticRegressionCV()
clf.fit(X, y)

plt.figure(1)
plt.subplot(2,1,1)
plot_decision_boundary(lambda x: clf.predict(x), X, y)


# Building a neural network
hdim = 4
model = build_model(nn_input_dim=2, nn_hdim=hdim, nn_output_dim=2,
                    X=X, y=y, num_examples=len(X),
                    reg_lambda=0.01, epsilon=0.01,
                    num_passes=20000)

print("Otrzymany model SNN: ", model)

# Plot the decision boundary
plt.subplot(2,1,2)
plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.title("Decision Boundary for hidden layer size " + str(hdim))
plt.show()




