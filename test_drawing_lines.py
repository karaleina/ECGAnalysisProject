from matplotlib import pyplot as plt
import numpy as np

list_of_hidden_neurons = [1,2]
one_minus_specifities = np.array([[0, 0.2,0.3,0.8, 1],[0, 0.5,0.6,0.95,1]])
sensitivities = np.array([[0, 0.7,0.8,0.9, 1],[0, 0.9,0.8,0.95, 1]])
plt.figure(1)

for index in range(len(list_of_hidden_neurons)):
    plt.plot(list(one_minus_specifities[index].ravel()), list(sensitivities[index].ravel()), linestyle='-',
             label="hidden neurons=" + str(int(list_of_hidden_neurons[index])))
plt.legend()
plt.grid()

plt.plot([0, 1], [0, 1], "m")
plt.axis([0, 1, 0, 1])
plt.ylabel("sensitivity", fontweight="bold")
plt.xlabel("1-specifity", fontweight="bold")
plt.title("ROC curve", fontweight="bold")
plt.show()