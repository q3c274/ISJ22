import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

"""
Code to reproduce the figure 3 of the paper.
For multiple levels of sex imbalance, we plot the mean ATE and the 25%/75%
quantiles across random deletionsof men or women in each table. We compare
matching & averaging, fuzzy matching, and embeddings with machine-learning.

"""

### Load experimental results

def load_data(model, q=0.75):
    p_list = [30, 55, 75, 88, 97, 100]
    ATE_0 = np.loadtxt(f'../results/ATE_depleted/{model}_0%.csv')[0]
    ATE_list = [np.loadtxt(f'../results/ATE_depleted/{model}_{p}%.csv')[:,0] for p in p_list]
    x = np.array([0] + p_list)
    y_med, q_sup, q_inf = [ATE_0], [ATE_0], [ATE_0]
    for ATEs in ATE_list:
        mask = ~np.isnan(ATEs) & (ATEs != np.inf)
        ATEs = ATEs[mask]
        y_med.append(np.mean(ATEs))
        q_sup.append(np.quantile(ATEs, q))
        q_inf.append(np.quantile(ATEs, 1-q))
    return x, y_med, q_sup, q_inf

def load_data_corrupted(model, q=0.75):
    p_list = [0, 30, 55, 75, 88, 97, 100]
    ATE_list = [np.loadtxt(f'../results/ATE_depleted/{model}_{p}%.csv')[:,0] for p in p_list]
    ATE_list = [ATEs[~np.isnan(ATEs)] for ATEs in ATE_list]
    y_med = np.array([np.mean(ATEs) for ATEs in ATE_list])
    q_sup = np.array([np.quantile(ATEs, q) for ATEs in ATE_list])
    q_inf = np.array([np.quantile(ATEs, 1-q) for ATEs in ATE_list])
    return p_list, y_med, q_sup, q_inf
    
x, ym_med, qm_sup, qm_inf = load_data('matching')
_, yl_med, ql_sup, ql_inf = load_data('learning')
_, ylt_med, qlt_sup, qlt_inf = load_data('learning_tuned')
_, yf_med, qf_sup, qf_inf = load_data('fuzzy')
_, ylc_med, qlc_sup, qlc_inf = load_data_corrupted('learning_corrupted')
_, yfc_med, qfc_sup, qfc_inf = load_data_corrupted('fuzzy_corrupted')
x_scaled = (1 - (1 - x/100)**0.5) * 100

### Plot ATE estimates quantiles

plt.rcParams['xtick.major.pad'] = '2'
plt.figure(dpi=180, figsize=(.79*3.5, .79*3))
plt.plot(x_scaled, ym_med, lw=2, color='C0', label='Matching & averaging')
plt.plot(x_scaled, yf_med, lw=2, color='C2', label='Fuzzy matching')
plt.plot(x_scaled, ylt_med, lw=2, color='C1', label='Embedding & learning')
plt.fill_between(x_scaled, qm_sup, qm_inf, color='C0', alpha=0.3)
plt.fill_between(x_scaled, qf_sup, qf_inf, color='C2', alpha=0.3)
plt.fill_between(x_scaled, qlt_sup, qlt_inf, color='C1', alpha=0.3)


plt.xlabel('Sex imbalance         ')
xtickslabels = ['0%\n(original data)'] + [f'{xk}%' for xk in x[1:-1]] + []

plt.xticks(x_scaled, labels=[f'{xk}%' for xk in x], ha='center', size=9)

ax = plt.gca()
plt.text(-0.1, -.22, 'original\ndata', transform=ax.transAxes, size=8)
plt.text(1.06, -.29, 'databases\ncontain only\nmen or women', transform=ax.transAxes, size=8, ha='right')
plt.xlim(x_scaled[0], x_scaled[-1])
plt.ylim(-5800, 13000)

plt.ylabel('Salary gap across sex ($)')
yticks = [-5000, 0, 5000, 10000]
plt.yticks(yticks, labels=[f'{x}' for x in yticks], va='center', rotation=90, size=9)

plt.legend(loc="lower left", frameon=True, handlelength=1, handletextpad=.5, borderaxespad=.2, borderpad=.3)
plt.tight_layout(pad=.01)
plt.savefig(f'../latex/figures/ate_depletion.pdf')
plt.show()

# def merge_chunks(model, n_chunks, p):
#     arr_list = []
#     for k in range(n_chunks):
#         arr_list.append(np.loadtxt(f'ATE_depleted/chunks/{model}_{p}%_{k}.csv'))
#     arr_merged = np.vstack(arr_list)
#     np.savetxt(fname=f'../results/ATE_depleted/{model}_{p}%.csv', X=arr_merged)
#     return

# merge_chunks('learning', 15, 30)
# merge_chunks('learning', 10, 55)
# merge_chunks('learning', 10, 75)
# merge_chunks('learning', 10, 88)
# merge_chunks('learning', 20, 97)

# merge_chunks('fuzzy', 7, 100)
# merge_chunks('fuzzy', 7, 97)
# merge_chunks('fuzzy', 7, 88)
# merge_chunks('fuzzy', 7, 75)
# merge_chunks('fuzzy', 7, 55)
# merge_chunks('fuzzy', 11, 30)

# merge_chunks('learning_corrupted', 11, 100)
# merge_chunks('learning_corrupted', 11, 97)
# merge_chunks('learning_corrupted', 11, 88)
# merge_chunks('learning_corrupted', 11, 75)
# merge_chunks('learning_corrupted', 11, 55)
# merge_chunks('learning_corrupted', 11, 30)
# merge_chunks('learning_corrupted', 11, 0)

# merge_chunks('fuzzy_corrupted', 12, 100)
# merge_chunks('fuzzy_corrupted', 12, 97)
# merge_chunks('fuzzy_corrupted', 12, 88)
# merge_chunks('fuzzy_corrupted', 12, 75)
# merge_chunks('fuzzy_corrupted', 8, 55)
# merge_chunks('fuzzy_corrupted', 6, 30)
# merge_chunks('fuzzy_corrupted', 6, 0)

# merge_chunks('learning_tuned', 6, 100)
# merge_chunks('learning_tuned', 11, 97)
# merge_chunks('learning_tuned', 12, 88)
# merge_chunks('learning_tuned', 8, 75)
# merge_chunks('learning_tuned', 8, 55)
# merge_chunks('learning_tuned', 11, 30)