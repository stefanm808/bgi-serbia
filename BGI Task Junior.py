### Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder

### Load the data
cols_to_load = ['geneID', 'cell', 'x', 'y', 'MIDCounts']
df = pd.read_csv('/kaggle/input/e145-e1s3-dorsal-midbrain-gem-cellbin-mergetsv/E14.5_E1S3_Dorsal_Midbrain_GEM_CellBin_merge.tsv', 
                 delimiter='\t', 
                 usecols=cols_to_load)

### Simply to check whether all the desired columns are displayed
df.columns

### To reduce the amount of data to a smaller subset in order to not timeout and exceed our CPU
df_subset = df.sample(n=1000, random_state=42)

### Performance of one-hot encoding on the categorical data
cat_vars = ['cell']
X_cat = pd.get_dummies(df_subset[cat_vars])
enc = OneHotEncoder()
X_cat_enc = enc.fit_transform(X_cat)

### To concatenate the given encoded categorical data with the numerical data which we need in order for PCA to work
num_vars = ['x', 'y', 'MIDCounts', 'cell']
num_vars_clean = [col.strip() for col in num_vars]
X_num = df_subset[num_vars_clean].values
X = pd.concat([pd.DataFrame(X_num), pd.DataFrame(X_cat_enc.toarray())], axis=1)

### Perform PCA on the preprocessed gene expression data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

### Combine the spatial coordinates and the 2D projection of the gene expression data
X_combined = np.hstack((X_num, X_pca))

### Perform spatial clustering on the combined data
db = DBSCAN(eps=100, min_samples=5).fit(X_combined)
labels = db.labels_

### Visualize the clustering results in 2D
sns.scatterplot(x=X_combined[:, 0], y=X_combined[:, 1], hue=labels, palette=sns.color_palette("hls", len(set(labels))))
plt.show()

### Extra graphing ~ visualize the clustering results in 3D, using matplotlib 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_combined[:, 0], X_combined[:, 1], X_combined[:, 1], c=labels, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()