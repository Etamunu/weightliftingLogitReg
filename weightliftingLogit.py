import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#######Load and format the data##################################
data = pd.read_csv('dataset.csv') # put the location of the dataset on your computer
data['hips'] = data['hips'].replace({'yes': 1, 'no': 0}) # correct the data format
data['sex'] = data['sex'].replace({'f': 0, 'm': 1}) # correct the data format
data['injury'] = data[['shoulder', 'knees', 'back', 'wrist', 'hips']].max(axis=1) # regroup all types of injury
data['train_total'] = data['train_days']*(data['train_lift']+data['train_strength']+data['train_supp'])/15 # measure of the training volume
data['age_dec'] = data['age']/10 # compute the age (in decades)
data['age_start_dec'] = data['age_start']/10 # compute the start age (in decades)
data['sport0'] = data[['sport0_power','sport0_body','sport0_cf','sport0_ball','sport0_fit','sport0_endure','sport0_track',\
    'sport0_ma','sport0_yoga','sport0_gym','sport0_strength','sport0_impact']].max(axis=1) # prior sport
data['pa'] = data[['pa_power','pa_body','pa_cf','pa_ball','pa_fit','pa_endure','pa_track','pa_ma','pa_yoga']].max(axis=1) # concurrent sport

# Select response variable and covariates
X = data[['sex', 'OA', 'age_dec', 'age_start_dec', 'nutrition','train_total', 'pown', 'pa','sport0']]
X = sm.add_constant(X)  # Adds a constant (intercept) term to the model
y = data['wrist'] # Select type of injury

########LOGISTIC REGRESSION##################################

# Fit the logistic regression model
model = sm.Logit(y, X)
result = model.fit()

# Print the summary
print(result.summary())

# coefficients and 95% confidence interval
coef = result.params
conf = result.conf_int()
conf.columns = ['2.5%', '97.5%']
conf['Coef'] = coef

conf = conf.loc[conf.index != 'const']

fig, ax = plt.subplots(figsize=(10,4))

positive_conf = conf[conf['2.5%'] > 0]
negative_conf = conf[conf['97.5%'] < 0]

# Plotting positive (red) error bars
ax.errorbar(positive_conf.index, positive_conf['Coef'],
            yerr=np.abs(positive_conf[['2.5%', '97.5%']].subtract(positive_conf['Coef'], axis=0)).values.T,
            fmt='o', color='red', ecolor='red', lw=3, capsize=7)

# Plotting negative (green) error bars
ax.errorbar(negative_conf.index, negative_conf['Coef'],
            yerr=np.abs(negative_conf[['2.5%', '97.5%']].subtract(negative_conf['Coef'], axis=0)).values.T,
            fmt='o', color='green', ecolor='green', lw=3, capsize=7)

# Adding neutral (black) error bars for the rest of the data
neutral_conf = conf.drop(positive_conf.index).drop(negative_conf.index)
ax.errorbar(neutral_conf.index, neutral_conf['Coef'],
            yerr=np.abs(neutral_conf[['2.5%', '97.5%']].subtract(neutral_conf['Coef'], axis=0)).values.T,
            fmt='o', color='black', ecolor='black', lw=3, capsize=7)

ax.set_title('Logistic Regression Coefficients (wrist injuries)', fontsize=16)
plt.ylabel('Coefficient', fontsize=16)
plt.grid(False)
ax.axhline(0, color='black', lw=1, linestyle='dashed')
plt.yticks(fontsize=14)
plt.xticks(rotation=45, fontsize=14)
plt.tight_layout()
plt.show()

# calculate the odds ratios
odds_ratios = np.exp(result.params)
print('Odds : ')
print(odds_ratios)

######## Check multicollinearity##################
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

print(vif)

# Compute and print the correlation matrix
import seaborn as sns

corr = X.corr()
f, ax = plt.subplots(figsize=(11, 9))
mask = np.tril(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
