#!/usr/bin/env python
# coding: utf-8

# In[32]:


import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# In[33]:


# Directory containing the audio files
dir_path = 'audios/audios'

# List of feature arrays
all_features = []

spectral_centroids_all = []
mfccs_all = []


# In[34]:


# Loop over all files
for file_name in os.listdir(dir_path):
    # Load the audio file
    audio, sr = librosa.load(os.path.join(dir_path, file_name), sr=None)

    # Calculate the spectrogram
    spectrogram = librosa.stft(audio)

    # Extract the spectral centroid feature
    spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(spectrogram), sr=sr)
    spectral_centroids_all.append(spectral_centroids)

    # Extract MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mels=40)
    mfccs_all.append(mfccs)

    # Concatenate the features
    features = np.concatenate((spectral_centroids, mfccs), axis=0)

    # Flatten the 2D features array to 1D before appending to the list
    all_features.append(features.T.flatten())



# In[35]:


# Convert list to numpy array
all_features = np.array(all_features)

scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)


# Dimensionality reduction with PCA
pca = PCA(n_components=10)
all_features = pca.fit_transform(all_features)


# In[36]:


# Clustering with KMeans
kmeans = KMeans(n_clusters=2, n_init=10)
labels = kmeans.fit_predict(all_features)


# In[40]:


# Analyze clustering results1
# 10-dimensional feature vector (after PCA) for each sample in each cluster.
for label in np.unique(labels):
    cluster_samples = all_features[labels == label]
    # Perform further analysis or visualization on each cluster
    # Example: Plotting the spectral centroid feature for each cluster
    plt.figure()
    for sample in cluster_samples:
        plt.plot(sample)
    plt.title(f'Cluster {label}')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.show()
    
# Analyze clustering results2
for label in np.unique(labels):
    
    # Compute average spectral centroid for this cluster
    avg_spectral_centroid = np.mean([spectral_centroids_all[i] for i in range(len(spectral_centroids_all)) if labels[i] == label], axis=0)
    
    # Plot average spectral centroid
    plt.figure()
    plt.plot(avg_spectral_centroid)
    plt.title(f'Cluster {label} - Average Spectral Centroid')
    plt.xlabel('Time')
    plt.ylabel('Magnitude')
    plt.show()


# In[18]:





# In[ ]:




