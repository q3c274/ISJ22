import numpy as np
from matplotlib import pyplot as plt

"""
Code to plot the figure 2 of the paper.

"""

### Load results
import pickle
def load_obj(name):
    with open(f'../results/variants_heterogeneity/{name}.pkl', 'rb') as f:
        return pickle.load(f)

results = load_obj('results')
results2 = load_obj('results2')
results3 = load_obj('results3')
results4 = load_obj('results4')

x_all = load_obj('x_all')
x2_all = load_obj('x2_all')
x3_all = load_obj('x3_all')
x4_all = load_obj('x4_all')

y_all = load_obj('y_all')
y2_all = load_obj('y2_all')
y3_all = load_obj('y3_all')
y4_all = load_obj('y4_all')

variants = ['9020s administrative assistant salary ',
            'administrative assistant',
            'administrative asst',
            'asst administrative']

variants2 = ['0361 project manager',
             '2128 project manager',
             '9109 project manager',
             'manager project',
             'mgr project',
             'project manager']

### PLOT
import seaborn as sns
sns.set()

plt.subplots(dpi=180, figsize=(3,2.8))

plt.subplot(221)
for v in variants2:
    x, counts, y = results3[v]
    y = np.array(y)
    mask = (x < 25)
    if v == 'project manager':
        plt.plot(x[mask], y[mask], linewidth=2, color='magenta')
    else:
        plt.plot(x[mask], y[mask], linewidth=0.8)
mask = (x3_all < 25)
plt.plot(x3_all[mask], np.array(y3_all)[mask], linewidth=2, label='empirical average', color='k')
plt.title('Matching & Averaging   ', size=8, pad=2)
ax = plt.gca()
plt.text(0, 1.13, "Classic approach", transform=ax.transAxes, size=10)
plt.text(-1.3, 1.13, "Query", transform=ax.transAxes, size=10)
plt.xticks([0, 10, 20, 30], ['']*4)
plt.xlim([0, 33])
plt.tick_params(axis='y', which='major', labelsize=8, rotation=90, pad=-5)
plt.yticks([80000, 120000, 160000], ['80k', '120k', '160k'], va='center')
plt.ylim([55000, 185000])
plt.ylabel('Annual Salary', size=8, labelpad=2)

plt.subplot(223)
for v in variants:
    x, counts, y = results[v]
    y = np.array(y)
    mask = (x < 34)
    if v == 'administrative assistant':
        plt.plot(x[mask], y[mask], linewidth=2, color='magenta')
    else:
        plt.plot(x[mask], y[mask], linewidth=0.8)
mask = (x_all < 34)
plt.plot(x_all[mask], np.array(y_all)[mask], linewidth=2, label='empirical average', color='k')
plt.xlabel('Years of experience', size=8, labelpad=2)
plt.ylim([30000, 60000])
plt.tick_params(axis='y', which='major', labelsize=8, rotation=90, pad=-5)
plt.tick_params(axis='x', which='major', labelsize=8, pad=-5)
plt.yticks([35000, 45000, 55000], ['35k', '45k', '55k'], va='center')
plt.ylabel('Annual Salary', size=8, labelpad=2)
plt.xlim([0, 33])
plt.xticks([0, 10, 20, 30], size=8)

plt.subplot(222)
for v in variants2:
    x, y = results4[v]
    mask = x < 25
#     if ' project manager' in v:
#         v = v.replace('project manager', 'project\nmanager')
    if v == 'project manager':
        plt.plot(x, y, label=f'"{v}"', linewidth=2, color='magenta')
    else:
        plt.plot(x, y, label=f'"{v}"', linewidth=0.8)
mask = (x4_all < 25)
plt.plot(x4_all, y4_all, linewidth=2, label='mean estimate', color='k')
plt.title('   Embedding & Learning', size=8, pad=2)
ax = plt.gca()
plt.text(0.2, 1.13, "Proposed", transform=ax.transAxes, size=10)
#plt.tick_params(axis='y', which='major', labelsize=10, rotation=90)
plt.yticks([80000, 120000, 160000], ['']*3, va='center')
plt.ylim([55000, 185000])
leg = plt.legend(bbox_to_anchor=(-1.28,1.12), ncol=1, fontsize=8, labelspacing=0.3, title='Project manager')
plt.setp(leg.get_title(),fontsize=9)
plt.xticks([0, 10, 20, 30], ['']*4)
plt.xlim([0, 33])

plt.subplot(224)
for v in variants:
    x, y = results2[v]
    if v == '9020s administrative assistant salary ':
        v = '9020s administrative\nassistant salary'
    if v == 'administrative assistant':
        v = 'administrative\nassistant'
    if v == 'administrative\nassistant':
        plt.plot(x, y, label=f'"{v}"', linewidth=2, color='magenta')
    else:
        plt.plot(x, y, label=f'"{v}"', linewidth=0.8)
mask = (x_all < 34)
plt.plot(x2_all, y2_all, linewidth=2, label='mean estimate', color='k')
plt.xlabel('Years of experience', size=8, labelpad=2)
plt.ylim([30000, 60000])
#plt.tick_params(axis='y', which='major', labelsize=8, rotation=90)
plt.yticks([35000, 45000, 55000], ['']*3, va='center')
leg = plt.legend(bbox_to_anchor=(-1.28, .97), ncol=1, fontsize=8, labelspacing=0.3,
           title='Administrative assistant')
plt.setp(leg.get_title(),fontsize=9)
plt.xlim([0, 33])
plt.xticks([0, 10, 20, 30], size=8)
plt.tick_params(axis='x', which='major', labelsize=8, pad=-5)

plt.subplots_adjust(wspace=0.04, hspace=0.05)
plt.savefig('../latex/figures/variants_dispersion.pdf',
            bbox_inches='tight', pad_inches=0.01)
plt.show()
