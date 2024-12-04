# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# analyze data type
df.dtypes
df.select_dtypes('object').head()

# process data
df = df.drop('MS_enc', axis=1)
df['MIF'] = pd.to_numeric(df['MIF'], errors='coerce')
df['SMIF'] = df['SMIF'].replace({'big': 0, 'bigger': 1, 'the_biggest': 2})

# find and remove outliers
z_scores = df.select_dtypes(np.number).apply(zscore)
threshold = 10
outliers = (np.abs(z_scores) > threshold)
df_outlier_removed = df[outliers.sum(axis=1)==0]
outliers.sum().sort_values(ascending=False)  # outliers count

df_melted = z_scores[['StCH', 'AATSC0v', 'n6Ring']].melt(var_name='feature', value_name='value')
sns.boxplot(x='feature', y='value', data=df_melted)
plt.title('Checking Outliers')

# check correlations
correlation_matrix = df_outlier_removed.corr()
n_feature = correlation_matrix.shape[0]
correlation_matrix = correlation_matrix - np.identity(n_feature)  # remove trivial pairs
corr_pairs = pd.DataFrame(correlation_matrix.unstack(), columns=['correlation'])
corr_pairs[abs(corr_pairs['correlation'])>0.999]

corr_pairs[abs(corr_pairs['correlation'])>=1]

# check correlation with the target variable
correlation_matrix['experimental_proprty'].abs().sort_values(ascending=False)

# plot correlation with the target variable
sns.scatterplot(
    data=df_outlier_removed,
    x='experimental_proprty',
    y='SLogP',
    s=10
)
plt.title('Most Correlated Feature to the Target Variable')

# process missing data
df_outlier_removed.columns[df_outlier_removed.isna().any()]

df_na_dropped = df_outlier_removed.drop(['MW2', 'MIF'], axis=1)

# seperate target from features and scale the data
features = df_na_dropped.drop(columns=['experimental_proprty'])
target = df_na_dropped['experimental_proprty']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# PCA
pca = PCA(0.95)
pca_result = pca.fit_transform(scaled_features)
len(pca.explained_variance_ratio_)

# PCA in 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['experimental_proprty'] = target
pca_df

# plot PCA
sns.scatterplot(
    data=pca_df,
    x='PC1',
    y='PC2',
    hue='experimental_proprty',
    palette="magma",
    s=15
)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA 2D Plot Colored by Target')
plt.show()

# tSNE
tsne = TSNE(n_components=2, perplexity=25)
tsne_result = tsne.fit_transform(scaled_features)
tsne_df = pd.DataFrame(data=tsne_result, columns=['Dim1', 'Dim2'])
tsne_df['experimental_proprty'] = target

# plot tSNE
sns.scatterplot(
    data=tsne_df,
    x='Dim1',
    y='Dim2',
    hue='experimental_proprty',
    palette='magma',
    s=20
)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE 2D Plot Colored by Target')
plt.show()

# train random forest regressor
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
rf_reg = RandomForestRegressor(n_estimators=100)
rf_reg.fit(X_train, y_train)

# evaluate the modal
y_pred = rf_reg.predict(X_test)

# visualize residuals
residuals = y_test - y_pred
sns.histplot(residuals, bins=30)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
