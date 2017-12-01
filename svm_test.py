from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt

X, y = make_classification(n_features=4, random_state=0)
clf = LinearSVC(random_state=0)
clf.fit(X, y)

# PARAMS
dual_problem = False #dual=False when n_samples > n_features.
class1 = 1 # only for plotting
class2 = 0 # only for plotting

#  SVM
LinearSVC(C=1.0, class_weight=None, dual=dual_problem, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,
     verbose=0)


new_y = clf.decision_function(X)
print(new_y)

# PLOTTING
plt.figure(1)
for index, class_y in enumerate(y):
    color = "blue" if class_y == 0 else "red"
    plt.scatter([X[index,class1]], X[index,class2], color=color )

# PLOTTING
plt.figure(2)
for index, class_y in enumerate(new_y):
    color = "blue" if class_y < 0 else "red"
    plt.scatter([X[index,class1]], X[index,class2], color=color )
plt.show()
