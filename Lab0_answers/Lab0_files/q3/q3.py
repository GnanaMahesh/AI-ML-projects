from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def func(t, v, k):
    """ computes the function S(t) with constants v and k """
    
    # TODO: return the given function S(t)
    l = v *(t- (1-np.exp(-k * t))/k)
    return l
    # END TODO


def find_constants(df: pd.DataFrame, func: Callable):
    """ returns the constants v and k """

    v = 0
    k = 0

    # TODO: fit a curve using SciPy to estimate v and k
    v,k = curve_fit(func,df['t'],df['S'])[0]
    # END TODO
    return v, k


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    v, k = find_constants(df, func)
    v = v.round(4)
    k = k.round(4)
    print(v, k)

    # TODO: plot a histogram and save to fit_curve.png
    output = func(df['t'],v,k)
    plt.plot(df['t'],output,color='red')
    plt.xlabel('t')
    plt.ylabel('S')
    plt.scatter(df['t'],df['S'],marker='*')
    name  = "fit: v="+str(v)+", k="+str(k)
    plt.legend([name,'data'])
    plt.savefig("fit_curve.png")
    # END TODO
