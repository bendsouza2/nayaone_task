import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import class_impl


pd.options.display.max_columns = 30

# Loading data
df = pd.read_csv('nayaone_loan_data 181122.csv')

# Data Profiling
print(df.head())
print(df.dtypes)
# ['verified_income', 'verification_income_joint', 'application_type', 'initial_listing_status', 'disbursement_method']
# are all object datatypes but can be converted to categorical variables to facilitate their use in the analysis


# Checking which columns contain categorical variables and their values
def categorical_variable(dataf, col):
    """Verify if a dataframe column is a categorical variable - defined here as any column for which all values fit into
    one of forty (or fewer) categories

    Returns a tuple containing the column name and a list of the unique values"""
    if len(dataf[col].unique()) <= 40:
        return col, list(dataf[col].unique())
    else:
        return False


def cat_var(dataf, col, switch=None):
    """Converts binary text columns to 1 and 0

    Args:
        dataf(pd.DataFrame): The dataframe
        col(str): The column name to apply the change to
        switch(dict)(Optional): The mappings for the binary values to switch (default=None)"""

    if switch is None:
        switch = {'yes': 1, 'no': 0}
    dataf[col] = dataf[col].str.lower()
    dataf[col] = dataf[col].map(switch)
    return dataf


cat_dict = {}
for column in list(df):
    if categorical_variable(df, column) is not False:
        col_name, unique_vals = categorical_variable(df, column)
        cat_dict[col_name] = unique_vals
print(cat_dict)

income_switch = {'not verified': 0, 'verified': 1, 'source verified': 2}
df = cat_var(df, 'verified_income', switch=income_switch)
df = cat_var(df, 'verification_income_joint', switch=income_switch)
df = cat_var(df, 'application_type', switch={'individual': 0, 'joint': 1})
df = cat_var(df, 'initial_listing_status', switch={'whole': 0, 'fractional': 1})
df = cat_var(df, 'disbursement_method', switch={'cash': 0, 'directpay': 1})

# Checking for data completeness
missing_vals = df.isnull().sum().to_frame(name='num_missing_vals')
missing_vals.drop(missing_vals[missing_vals.num_missing_vals == 0].index, inplace=True)
print(missing_vals)
# Insight
# emp_title and emp_length are the most problematic of the columns with missing values, since we would expect the
# columns that apply only to joint loan applications to be empty for people who applied to individual loans. We would
# also expect the columns around credit delinquency to be empty for people who have made all previous payments on time.
# We wouldn't expect the emp_title to be empty however. Although it sounds feasible that empty emp_title could
# correspond to unemployed people, there are varying annual incomes for people whose employment title is empty,
# suggesting that they are not all unemployed and that the data here is missing rather than not applicable.


# Checking for imbalance in the dataset
print(df['grade'].value_counts(normalize=True))
# Insight
# The dataset is imbalanced - the loans with high interest rates (riskier loans, E - G) are significantly
# underrepresented in the dataset.

# Visualising the data
lots_missing = list(missing_vals[(missing_vals['num_missing_vals'] > 1000)].index)
df_full = df.drop(lots_missing, axis=1)

plt.figure(figsize=(20, 20))
sns.set(font_scale=0.6)
sns.heatmap(df_full.corr(numeric_only=True), annot=True, cbar=False, cmap='Blues')
plt.title('Correlation Matrix')
plt.savefig(os.getcwd() + '/c_matrix.jpeg')
# Insight
# Strong positive correlation: [(tax_liens & num_historical failed to pay (0.87)),
#                               (total_credit_limit & annual_income (0.52)]

encoded = pd.get_dummies(df, columns=['grade'])

order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
plt.figure(figsize=(10, 10))
sns.boxplot(x=df['grade'], y=df['annual_income'], order=order, showfliers=False)
plt.title('Grade against annual_income boxplot')
plt.savefig(os.getcwd() + '/annual_income_boxp.jpeg')
# Insight
# We can see that for grades A - E a higher median income corresponds to a better loan grade (A). Groups F and G
# do not follow this trend, but it must be kept in mind that they are heavily undersampled in the dataset, which might
# skew the results for these loans.

plt.figure(figsize=(10, 10))
sns.boxplot(x=df['grade'], y=df['debt_to_income'], order=order, showfliers=False)
plt.title('Grade against debt_to_income boxplot')
plt.savefig(os.getcwd() + '/debt_to_income_boxp.jpeg')
# Insight
# Lower average debt to income ratio for the higher loan grades. Applicants with a lower debt to income ratio more
# likely to receive a better loan grade.

plt.figure(figsize=(10, 10))
sns.boxplot(x=df['grade'], y=df['account_never_delinq_percent'], order=order, showfliers=False)
plt.title('Grade against account_never_delinq_percent boxplot')
plt.savefig(os.getcwd() + '/account_never_delinq_boxp.jpeg')
# Insight
# Applicants with a higher percentage of credit against which they were never delinquent are more likely to receive a
# better loan grade. Underrepresented E-G buck the trend. Could use class weights or minority oversampling in model
# building to address this?

plt.figure(figsize=(10, 10))
sns.boxplot(x=df['grade'], y=df['total_credit_limit'], order=order, showfliers=False)
plt.title('Grade against total_credit_limit boxplot')
plt.savefig(os.getcwd() + '/total_credit_limit_boxp.jpeg')
plt.show()
# Insight
# A higher total credit limit (excluding mortgage) is associated with a better loan grade. Again, the underrepresented
# columns (E-G) do not follow the trend

dataset = class_impl.Dataset(os.getcwd() + '/nayaone_loan_data 181122.csv')
print(dataset.all_outliers())
