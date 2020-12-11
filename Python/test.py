import pandas as pd 
import matplotlib.image as mpimg
from IPython.display import Image, display
from functools import reduce
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


def plot_comparison(table, alpha):
    order_1 = table[table.Order.eq(1.0) & table.Alpha.eq(alpha)]
    order_2 = table[table.Order.eq(2.0) & table.Alpha.eq(alpha)]
    order_3 = table[table.Order.eq(3.0) & table.Alpha.eq(alpha)]
    order_4 = table[table.Order.eq(4.0) & table.Alpha.eq(alpha)]


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))

    for name, group in order_1.groupby('Type'):
        group.plot(x='Mesh Size', y='Error', ax=ax1, label=name, legend=True, title='k = 1' + ' alpha=' + str(alpha), style='.-', loglog=True)

    for name, group in order_2.groupby('Type'):
        group.plot(x='Mesh Size', y='Error', ax=ax2, label=name, legend=True,title='k = 2' + ' alpha=' + str(alpha), style='.-', loglog=True)

    for name, group in order_3.groupby('Type'):
        group.plot(x='Mesh Size', y='Error', ax=ax3, label=name, legend=True,title='k = 3' + ' alpha=' + str(alpha), style='.-', loglog=True)

    for name, group in order_4.groupby('Type'):
        group.plot(x='Mesh Size', y='Error', ax=ax4, label=name, legend=True,title='k = 4'+ ' alpha=' + str(alpha), style='.-', loglog=True)

    plt.savefig('images/dg_alpha_' + str(alpha)) 



results = pd.read_csv('nonsymmetric_results.csv', header='infer', index_col=0)
alphas = range(5, 105, 5)
for value in alphas:
    plot_comparison(results, value)
    print('...............................')