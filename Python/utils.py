import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import pandas as pd
from functools import reduce
from IPython.display import Image, display
import matplotlib.image as mpimg

# TODO: Modularize
def plot_error(table):
    order_1 = table[table.Order.eq(1.0)]
    order_2 = table[table.Order.eq(2.0)]
    order_3 = table[table.Order.eq(3.0)]
    order_4 = table[table.Order.eq(4.0)]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,5))

    order_1.plot(x='Mesh Size', y='Error', ax=ax1, legend=True, title='k = 1', style='.-', loglog=True)
    order_2.plot(x='Mesh Size', y='Error', ax=ax2, legend=True, title='k = 2', style='.-', loglog=True)
    order_3.plot(x='Mesh Size', y='Error', ax=ax3, legend=True, title='k = 3', style='.-', loglog=True)
    order_4.plot(x='Mesh Size', y='Error', ax=ax4, legend=True, title='k = 4', style='.-', loglog=True)