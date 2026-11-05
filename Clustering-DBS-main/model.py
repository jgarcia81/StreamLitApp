import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
#from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

def preprocessing(df, features):
    X = df[features].copy()

    numerical = X.select_dtypes(include=['int64', 'float64']).columns
    categorical = X.select_dtypes(include=['object', 'category']).columns

    scaler = StandardScaler()
    X[numerical] = scaler.fit_transform(X[numerical])
    
    encoder = LabelEncoder()
    X[categorical] = X[categorical].apply(encoder.fit_transform)

    return X

def kmeans(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model.labels_

def kmedoids(X, n_clusters, metric):
    model = KMedoids(n_clusters=n_clusters, metric=metric, random_state=42)
    model.fit(X)
    return model.labels_


def clustering(X, n_clusters, linkage):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    model.fit(X)
    return model.labels_

def dbscan(X, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model.labels_

def evaluate(X, labels):
    unique_labels = set(labels) - {-1}
    if len(unique_labels) < 2:
        return None, None
    return silhouette_score(X, labels), davies_bouldin_score(X, labels)

def apply_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    columns = [f'PCA_{i+1}' for i in range(n_components)]
    return pd.DataFrame(X_pca, columns=columns)