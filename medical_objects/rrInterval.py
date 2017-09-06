from matplotlib import pyplot as plt

class RRInterval(object):

    def __init__(self, interval=[]):
        self.__interval = interval

    def plot_interval(self):
        plt.figure(1)
        plt.plot(self.__interval)
        plt.show()




