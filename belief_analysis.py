import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# dataset 
datafews = pd.read_csv(r'C:\Users\jbrittes\Documents\GitHub\datafewsion_food\2023-11-13_All_Data_from_2018.csv')


# economic and agricultural variables
economic_agricultural_vars = [col for col in datafews.columns if col not in [
    'affectweather', 'affectweatherOppose', 'consensus', 'consensusOppose', 
    'devharm', 'devharmOppose', 'discuss', 'discussOppose', 'futuregen', 'futuregenOppose', 
    'happening', 'happeningOppose', 'harmplants', 'harmplantsOppose', 'harmUS', 
    'GEOID', 'County', 'Year'  # Excluding identifiers and year
]]


# --------------------- diagonal correlation matrix -------------------------------------
sns.set_theme(style='white')

d = datafews.drop(columns = {'Year', 'County', "GEOID"}, axis = 1)
corr = d.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

f, ax = plt.subplots(figsize = (11,9))

cmap = sns.diverging_palette(230,20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, center = 0, square = True, linewidths=0.5, cbar_kws={'shrink': 0.5})


# ---------------------- Summary / very slow----------------------------------------------
d = datafews.drop(columns = {'Year', "GEOID", 'affectweather', 'affectweatherOppose', 'consensus', 'consensusOppose', 
    'devharm', 'devharmOppose', 'discuss', 'discussOppose', 'futuregen', 'futuregenOppose', 
    'happeningOppose', 'harmplants', 'harmplantsOppose', 'harmUS'}, axis = 1)
sns.pairplot(d, hue = 'County')


# ----------------- Individual Regression Analysis ----------------------------------------
import statsmodels.api as sm

# Selecting variables for the analysis
variables_to_analyze = [
    ('affectweather', 'Real_GDP_Thousands_of_Chained_2012_dollars_2018'),
    ('consensus', 'Beef_cattle_head'),
    ('happening', 'Corn_yield_bu_ac')
]

# Performing linear regression for each combination
regression_results = {}

for dependent_var, independent_var in variables_to_analyze:
    X = sm.add_constant(datafews[independent_var])  # Adding a constant term
    Y = datafews[dependent_var]

    model = sm.OLS(Y, X).fit()
    regression_results[(dependent_var, independent_var)] = model.summary()

regression_results

# Visualize
for y_var, x_var in variables_to_analyze:
    X = sm.add_constant(datafews[x_var])
    Y = datafews[y_var]
    model = sm.OLS(Y, X).fit()
    
    plt.figure(figsize=(8, 6))
    plt.scatter(datafews[x_var], Y, alpha=0.5)
    plt.plot(datafews[x_var], model.predict(X), color='red')
    plt.title(f'Regression of {y_var} on {x_var}')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.show()


# ----------------- Multiple Regression Analysis ----------------------------------------
dependent_var = 'happening'  # Example public opinion variable
independent_vars = ['Real_GDP_Thousands_of_Chained_2012_dollars_2018', 'Beef_cattle_head', 'Corn_yield_bu_ac']  # Example independent variables

# Preparing the data for regression
X = sm.add_constant(datafews[independent_vars])  # Adding a constant term
Y = datafews[dependent_var]

# Performing multiple regression
model = sm.OLS(Y, X).fit()

#print(model.summary())

# Visual
from statsmodels.graphics.regressionplots import plot_partregress_grid

# Partial regression plots
fig = plt.figure(figsize=(12, 8))
plot_partregress_grid(model, fig=fig)
plt.show()


dependent_var = 'happening'  # Example public opinion variable
independent_vars = ['Real_GDP_Thousands_of_Chained_2012_dollars_2018', 'Beef_cattle_head', 'Corn_yield_bu_ac']  # Example independent variables

# Preparing the data for regression
X = sm.add_constant(datafews[independent_vars])  # Adding a constant term
Y = datafews[dependent_var]

# Performing multiple regression
model = sm.OLS(Y, X).fit()

#print(model.summary())

# Visual
from statsmodels.graphics.regressionplots import plot_partregress_grid

# Partial regression plots
fig = plt.figure(figsize=(12, 8))
plot_partregress_grid(model, fig=fig)
plt.show()


# ------------------------------------------------------------------------------------------
