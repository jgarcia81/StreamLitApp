import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, kmeans, kmedoids, clustering, dbscan, evaluate, apply_pca

st.title('Clustering App')

file = st.file_uploader('upload your dataset', type=['csv'])

if file is not None:
    df = pd.read_csv(file)
else:
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())


feature_mode = st.radio('Select feature mode:', ['Manual Selection', 'PCA Reduction'])

if feature_mode == 'Manual Selection':
    features = st.multiselect('Select the features for clustering (at least 2)', df.columns.to_list())
    if len(features) < 2:
        st.warning('Select at least 2 features')
        st.stop()
    X = preprocessing(df, features)

else:  # PCA 
    all_features = st.multiselect('Select features to include in PCA (leave empty for all)', df.columns.to_list(), default=df.columns.to_list())
    if len(all_features) < 2:
        st.warning('Select at least 2 features to perform PCA')
        st.stop()
    
    n_components = st.number_input('Number of PCA components', min_value=2, max_value=min(len(all_features), 10), value=2)
    X_pre = preprocessing(df, all_features)
    X = apply_pca(X_pre, n_components=n_components)
    features = X.columns.to_list()

#Model Selection
model = st.selectbox('Select model', ['K-Means', 'K-Medoids', 'Hierarchical Clustering', 'DBSCAN'])

#Hyperparameters
if model == 'K-Means':
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    labels = kmeans(X, n_clusters=n_clusters)

elif model == 'K-Medoids':
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    metric = st.selectbox('Select metric', ['euclidean', 'manhattan', 'cosine'])
    labels = kmedoids(X, n_clusters=n_clusters, metric=metric)

elif model == 'Hierarchical Clustering':
    n_clusters = st.slider('Number of clusters', 2, 10, 3)
    linkage = st.selectbox('Select linkage', ['ward', 'single', 'complete', 'average'])
    labels = clustering(X, n_clusters=n_clusters, linkage=linkage)

elif model == 'DBSCAN':
    eps = st.number_input("Enter eps", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    min_samples = st.slider('Number of samples (min_samples)', 2, 20, 5)
    labels = dbscan(X, eps=eps, min_samples=min_samples)
    
    if len(set(labels) - {-1}) < 1:
        st.warning('The hyperparameters must be set. (Only 1 cluster or noise detected)')
        st.stop()

df['cluster'] = labels

st.subheader('Cluster Labels')
st.write(pd.Series(labels, name='cluster').value_counts())

st.subheader('Clustered Dataset')
st.dataframe(df)

silhoutte, davis = evaluate(X, labels)
if silhoutte is not None:
    st.write('Silhouette:', silhoutte)
    st.write('Davies-Bouldin Index:', davis)
else:
    st.warning('Could not calculate metrics: At least 2 clusters are required.')

st.subheader('Visualization')

if len(features) == 2:
    fig, ax = plt.subplots()
    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)

elif len(features) > 2:
    st.info('More than 2 features selected. Applying PCA to 2 components for 2D visualization.')
    X_viz = apply_pca(X, n_components=2)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_viz.iloc[:, 0], X_viz.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)