import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# ### Average Treatment Effect

# Load data
DB1_model = np.loadtxt('../results/ATE_subsets/model/1DB_matched=False.csv')[:,0]
DB1_model_rot = np.loadtxt('../results/ATE_subsets/model/1DB_matched=False_rotated.csv')[:,0]
DB1_emp_IPW = np.loadtxt('../results/ATE_subsets/empirical/1DB_empirical_IPW_dismiss=False.csv')
DB1_emp_IPW_dismiss = np.loadtxt('../results/ATE_subsets/empirical/1DB_empirical_IPW_dismiss=True.csv')
DB1_fuzzy = np.loadtxt('../results/ATE_subsets/fuzzy/1DB.csv')
DB1_fuzzy_rot = np.loadtxt('../results/ATE_subsets/fuzzy/1DB_rotated.csv')

DB3_model = np.loadtxt('../results/ATE_subsets/model/3DB_matched=False.csv')[:,0]
DB3_model_rot = np.loadtxt('../results/ATE_subsets/model/3DB_matched=False_rotated.csv')[:,0]
DB3_emp_IPW = np.loadtxt('../results/ATE_subsets/empirical/3DB_empirical_IPW_dismiss=False.csv')
DB3_emp_IPW_dismiss = np.loadtxt('../results/ATE_subsets/empirical/3DB_empirical_IPW_dismiss=True.csv')
DB3_fuzzy = np.loadtxt('../results/ATE_subsets/fuzzy/3DB.csv')
DB3_fuzzy_rot = np.loadtxt('../results/ATE_subsets/fuzzy/3DB_rotated.csv')

DB7_model = np.loadtxt('../results/ATE_subsets/model/7DB_matched=False.csv')[:,0]
DB7_model_rot = np.loadtxt('../results/ATE_subsets/model/7DB_matched=False_rotated.csv')[:,0]
DB7_emp_IPW = np.loadtxt('../results/ATE_subsets/empirical/7DB_empirical_IPW_dismiss=False.csv')
DB7_emp_IPW_dismiss = np.loadtxt('../results/ATE_subsets/empirical/7DB_empirical_IPW_dismiss=True.csv')
DB7_fuzzy = np.loadtxt('../results/ATE_subsets/fuzzy/7DB.csv')
DB7_fuzzy_rot = np.loadtxt('../results/ATE_subsets/fuzzy/7DB_rotated.csv')

DB14_model = np.loadtxt('../results/ATE_subsets/model/14DB_matched=False.csv')[:1]
DB14_model_rot = np.loadtxt('../results/ATE_subsets/model/14DB_matched=False_rotated.csv')[:1]
DB14_emp_IPW = np.loadtxt('../results/ATE_subsets/empirical/14DB_empirical_IPW_dismiss=False.csv')
DB14_emp_IPW_dismiss = np.loadtxt('../results/ATE_subsets/empirical/14DB_empirical_IPW_dismiss=True.csv')
DB14_fuzzy = np.loadtxt('../results/ATE_subsets/fuzzy/14DB.csv')
DB14_fuzzy_rot = np.loadtxt('../results/ATE_subsets/fuzzy/14DB_rotated.csv')

### Make dataframe
N = 6
df = pd.DataFrame(np.zeros((14*3*N+N, 3)), columns=['approach', 'subset_size', 'ATE'])
df['approach'] = ['Average treatment effect with embedding & learning,\non artificially mismatched data'] * (14*3+1) +['Average treatment effect with embedding & learning'] * (14*3+1) +['Average treatment effect with fuzzy matching,\non artificially mismatched data'] * (14*3+1) +['Average treatment effect with fuzzy matching'] * (14*3+1) +['Average treatment effect with matching & averaging'] * (14*3+1) +['Salary gap across sex for employees with same experience,\nethnicity, employer, and manually-cleaned job title (dropping 60%\nof the employees for which there is no opposite-sex correspondance)'] * (14*3+1)
df['subset_size'] = [1]*14 + [3]*14 + [7]*14 + [14] + [1]*14 + [3]*14 + [7]*14 + [14] +[1]*14 + [3]*14 + [7]*14 + [14] + [1]*14 + [3]*14 + [7]*14 + [14] + [1]*14 + [3]*14 + [7]*14 + [14]+ [1]*14 + [3]*14 + [7]*14 + [14]
df['ATE'] = np.concatenate([DB1_model_rot, DB3_model_rot, DB7_model_rot, DB14_model_rot,
                            DB1_model, DB3_model, DB7_model, DB14_model,
                            DB1_fuzzy_rot, DB3_fuzzy_rot, DB7_fuzzy_rot, np.array([DB14_fuzzy_rot]),
                            DB1_fuzzy, DB3_fuzzy, DB7_fuzzy, np.array([DB14_fuzzy]),
                            DB1_emp_IPW, DB3_emp_IPW, DB7_emp_IPW, np.array([DB14_emp_IPW]),
                            DB1_emp_IPW_dismiss, DB3_emp_IPW_dismiss, DB7_emp_IPW_dismiss,
                            np.array([DB14_emp_IPW_dismiss])])

### Make dataframe
N = 4
df = pd.DataFrame(np.zeros((14*3*N+N, 3)), columns=['approach', 'subset_size', 'ATE'])
df['approach'] = ['Average treatment effect with embedding & learning'] * (14*3+1) +['Average treatment effect with fuzzy matching'] * (14*3+1) +['Average treatment effect with matching & averaging'] * (14*3+1) +['Salary gap across sex for employees with same experience,\nethnicity, employer, and manually-cleaned job title (dropping 60%\nof the employees for which there is no opposite-sex correspondance)'] * (14*3+1)
df['subset_size'] = [1]*14 + [3]*14 + [7]*14 + [14] + [1]*14 + [3]*14 + [7]*14 + [14] +[1]*14 + [3]*14 + [7]*14 + [14] + [1]*14 + [3]*14 + [7]*14 + [14]
df['ATE'] = np.concatenate([DB1_model, DB3_model, DB7_model, DB14_model,
                            DB1_fuzzy, DB3_fuzzy, DB7_fuzzy, np.array([DB14_fuzzy]),
                            DB1_emp_IPW, DB3_emp_IPW, DB7_emp_IPW, np.array([DB14_emp_IPW]),
                            DB1_emp_IPW_dismiss, DB3_emp_IPW_dismiss, DB7_emp_IPW_dismiss,
                            np.array([DB14_emp_IPW_dismiss])])

fig, ax = plt.subplots(dpi=180, figsize=(4,3))
sns.boxplot(x='subset_size', y='ATE', hue='approach', data=df.iloc[::-1], whis=0, showfliers=False)
plt.xlabel('Number of databases', size=12)
plt.ylabel('Salary gap across sex', size=12)
#plt.hlines([3982], xmin=-0.5, xmax=3.5, linestyle='--', label='Only employees\nmatched across\nsex and covariates')
#plt.plot(3.33, DB14_model_rot, 'o', color='tab:brown')
plt.plot(3.3, DB14_model, 'o', color='tab:red')
#plt.plot(3.06, DB14_fuzzy_rot, 'o', color='tab:red')
plt.plot(3.1, DB14_fuzzy, 'o', color='tab:green')
plt.plot(2.9, DB14_emp_IPW, 'o', color='tab:orange')
plt.plot(2.7, DB14_emp_IPW_dismiss, 'o', color='tab:blue')
plt.legend(bbox_to_anchor=(1.6,-.2), fontsize=12)
plt.tick_params(axis='y', rotation=90)
yticks = [0, 5000, 10000]
ax.set_yticks(yticks)
ax.set_yticklabels(labels=[f'${x}' for x in yticks], va='center')
plt.savefig(f'../latex/figures/ate_subsets.pdf', bbox_inches='tight')
plt.show()