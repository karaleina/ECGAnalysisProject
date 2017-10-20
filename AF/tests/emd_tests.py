from PyEMD import EMD
import numpy as np
from matplotlib import pyplot as plt


s = np.random.random(100)
emd = EMD()
IMFs = emd.emd(s)

plt.figure(-1)
plt.plot(s)

for index, imf in enumerate(IMFs):
    plt.figure(index)
    plt.plot(imf)

plt.show()

