"""
Kernel regression model
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import pickle


class KernelRegression():
    """Nadaraya-Watson kernel regression"""
    def __init__(self, kernel='box'):
        """
        :param kernel: type of kernel: 
            one of {'box','epanechnikov','tri-cube','gaussian'}
        """
        self.kernel = kernel
        
    def fit(self, X, y):
        """
        Fit model: X and y are 1D arrays
        """
        self.X = X
        self.y = y
        
    def predict(self, bandwidth=1):
        """Get prediction from X using bandwidth"""
        y_pred = []
        for ii in range(len(self.X)):
            K = KERNEL_FUNC[self.kernel](self.X, ii, h=bandwidth)
            y_pred.append(np.sum(K*self.y, axis=0) / np.sum(K,axis=0))
        return np.array(y_pred)
    

# Define different kernel functions
def box_kernel(x, i, h=1):
    """
    Calculate box kernel
    x: array of values
    i: index of value to calculate kernel
    h: bandwidth
    Return: array of kernel values
    """
    # Calculate box kernel for now
    K = np.zeros_like(x, dtype=np.float)
    for v in range(i-h, i+h+1):
        if 0<= v < len(x):
            K[v] += 0.5
    return K


def epan_kernel(x, i, h=1):
    """
    Calculate Epanechnikov kernel
    x: array of values
    i: index of value to calculate kernel
    h: bandwidth
    Return: array of kernel values
    """
    K = np.zeros_like(x, dtype=np.float)
    for v in range(i-h, i+h+1):
        if 0<= v < len(x):
            v_ = (x[v]-x[i])/h
            K[v] += 0.75*(1-v_**2)
    return K


def tri_cube_kernel(x, i, h=1):
    """
    Calculate Tri-cube kernel
    x: array of values
    i: index of value to calculate kernel
    h: bandwidth
    Return: array of kernel values
    """
    K = np.zeros_like(x, dtype=np.float)
    for v in range(i-h, i+h+1):
        if 0<= v < len(x):
            v_ = (x[v]-x[i])/h
            K[v] += (70/81)*(1-np.abs(v_)**3)**3
    return K


def gauss_kernel(x, i, h=1):
    """
    Calculate Gaussian kernel
    x: array of values
    i: index of value to calculate kernel
    h: bandwidth
    Return: array of kernel values
    """
    K = np.zeros_like(x, dtype=np.float)
    for v in range(len(K)):
        K[v] = 1/(np.sqrt(2*np.pi))*np.exp(-(x[i]-x[v])**2/(h**2 *2))
    return K


KERNEL_FUNC = {
    'box': box_kernel,
    'epanechnikov': epan_kernel,
    'tri-cube': tri_cube_kernel,
    'gaussian': gauss_kernel
}


with open('data','rb') as f:
        data = pickle.load(f)

if __name__ == "__main__":
    # Load data
    df = pd.DataFrame(data['AAPL'])
    all_x = df.index.values
    all_y = df.Adj_Close.values
    # Fit model and output fitted data
    kr = KernelRegression(kernel='epanechnikov')
    kr.fit(all_x, all_y)
    y_kr = kr.predict(bandwidth=5)
    local_max = argrelextrema(y_kr, np.greater)[0]
    local_min = argrelextrema(y_kr, np.less)[0]
    plt.figure(figsize=(20,10),dpi=300)
    plt.plot(df.Adj_Close.values, zorder=1)
    plt.scatter(x=local_max,y=df.Adj_Close.values[local_max],s=5,c='g',zorder=2)
    plt.scatter(x=local_min,y=df.Adj_Close.values[local_min],s=5,c='r',zorder=2)
    plt.plot(y_kr)
    plt.savefig('output.jpg')
    print('Saved output to current directory!')