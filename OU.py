import random
import numpy as np


#Ornstein-Uhlenbeck process see: https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html

class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)