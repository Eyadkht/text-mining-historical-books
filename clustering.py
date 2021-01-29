# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 01:50:48 2020
@author: eyadk
"""
import pickle
import string

import re
import pandas as pd
import numpy as np
import nltk, sklearn
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_samples, silhouette_score
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import mpld3

#### What is the Optimal number of clusters
#### What is the correct parameters that will give the best performance

books_dic = {"gap_-C0BAAAAQAAJ":"Dictionary of Greek and Roman Geography",
                "gap_DhULAAAAYAAJ":"The Description of Greece",
                "gap_udEIAAAAQAAJ":"Pliny's Natural History",

                "gap_2X5KAAAAYAAJ":"The History of Tacitus",
                "gap_CSEUAAAAYAAJ":"The History of the Decline and Fall of the Roman Empire Vol III",
                "gap_GIt0HMhqjRgC":"The History of the Decline and Fall of the Roman Empire Vol IV",
                "gap_IlUMAQAAMAAJ":"The History of the Decline and Fall of the Roman Empire Vol II",
                "gap_VPENAAAAQAAJ":"The History of the Decline and Fall of the Roman Empire Vol V",
                "gap_XmqHlMECi6kC":"The History of the Decline and Fall of the Roman Empire Vol VI",
                "gap_aLcWAAAAQAAJ":"The History of the Decline and Fall of the Roman Empire Vol I",
                "gap_dIkBAAAAQAAJ":"The History of Rome Vol III",
                "gap_fnAMAAAAYAAJ":"The History of the Peloponnesian War Vol I",
                "gap_9ksIAAAAQAAJ":"The History of the Peloponnesian War Vol II",

                "gap_Bdw_AAAAYAAJ":"Livy History of Rome Vol I",
                "gap_m_6B1DkImIoC":"Livy History of Rome Vol II",
                "gap_DqQNAAAAYAAJ":"Livy History of Rome Vol III",
                "gap_RqMNAAAAYAAJ":"Livy History of Rome Vol V",

                "gap_TgpMAAAAYAAJ":"The Works of Flavius Josephus Vol I",
                "gap_CnnUAAAAMAAJ":"The Works of Flavius Josephus Vol II",
                "gap_y-AvAAAAYAAJ":"The Works of Flavius Josephus Vol III",
                "gap_ogsNAAAAIAAJ":"The Works of Flavius Josephus Vol IV",

                "gap_WORMAAAAYAAJ":"The Histories of Caius Cornelius Tacitus",
                "gap_pX5KAAAAYAAJ":"The Works of Cornelius Tacitus Vol IV",
                "gap_MEoWAAAAYAAJ":"The Historical Annals of Cornelius Tacitus Vol I",
                }

#################### Loading Files ##########################
f1 = open("tfidf_vectorizer.pkl","rb")
f2 = open("tfidf_matrix.pkl","rb")

documents = open("document_dictionary.pkl","rb")
vocabulary = open("total_vocab.pkl","rb")
doc_text = open("document_text.pkl","rb")

documents_dic = pickle.load(documents)
documents.close()

tfidf_vectorizer = pickle.load(f1)
tfidf_matrix = pickle.load(f2)

print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()

############################# Clustering Documents ###########################
from sklearn.cluster import KMeans

num_clusters = 6

km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

sse = []
list_k = list(range(1, 24))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(tfidf_matrix)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');

############################# Documents Similarity Heatmap ###########################
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances

dist = 1 - cosine_similarity(tfidf_matrix)
dist2 = 1- euclidean_distances(tfidf_matrix)

import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()

ax = sns.heatmap(dist, linewidths=.1)
ax4 = sns.heatmap(dist2, linewidths=.1)


####################################### Multidimensional Scaling #############################
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

#########################
books_titles = []

for doc in documents_dic.keys():
    text = doc.decode("utf-8")
    re.search("gap", text).start()
    word = ""
    for x in range(re.search("gap", text).start(),len(text)):
        word = word + text[x]
    books_titles.append(books_dic[word])

############################# Finding the best number of Clusters ###########################
for i, k in enumerate([2, 3, 4,5,6,7,8]):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    # Run the Kmeans algorithm
    km = KMeans(n_clusters=k)
    labels = km.fit_predict(tfidf_matrix)
    centroids = km.cluster_centers_
    clusters = km.labels_.tolist()
    # Get silhouette samples
    silhouette_vals = silhouette_samples(tfidf_matrix, labels)

    # Silhouette plot
    y_ticks = []
    y_lower, y_upper = 0, 0
    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_vals[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
        ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Get the average silhouette score and plot it
    avg_score = np.mean(silhouette_vals)
    print(i+1,"####",avg_score)
    ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
    ax1.set_yticks([])
    ax1.set_xlim([-0.1, 1])
    ax1.set_xlabel('Silhouette coefficient values')
    ax1.set_ylabel('Cluster labels')
    ax1.set_title('Silhouette plot for the various clusters', y=1.02)

    # Scatter plot of data colored with labels
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=books_titles))
    groups = df.groupby('label')
    for name, group in groups:
        ax2.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                mec='none')
        ax2.set_aspect('auto')
        ax2.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax2.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

        ax2.legend(numpoints=1)  #show legend with only 1 point
    ax2.set_title('Document Clusters Boundaries', y=1.02);
    # for i in range(len(df)):
    #     ax2.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)

############################# Clusters Vs Titles ###########################
books_titles = []

for doc in documents_dic.keys():
    text = doc.decode("utf-8")
    re.search("gap", text).start()
    word = ""
    for x in range(re.search("gap", text).start(),len(text)):
        word = word + text[x]
    books_titles.append(books_dic[word])


doc_vis2 = { 'document': books_titles, 'clusters': clusters }
frame = pd.DataFrame(doc_vis2, index = [clusters] , columns = ['document', 'clusters'])
frame.to_excel("frame.xlsx")
frame['clusters'].value_counts()

print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

cluster_titles=[]
cluster_words=[]
cluster_num=[]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    print()
    cluster_num.append(i+1)
    for ind in order_centroids[i, :10]: #replace 6 with n words per cluster
        print(terms[ind])

        cluster_words.append(terms[ind])
    print()

    print("Cluster %d titles:" % i, end='')
    for title in frame.loc[i]['document'].values.tolist():
        print(' %s,' % title, end='')
        cluster_titles.append(title)
    print() #add whitespace

doc_vis3 = { 'cluster': cluster_num, 'titles': cluster_titles , 'words':cluster_words}
df_cluster = pd.DataFrame(doc_vis3, columns = ['cluster', 'titles','words'])

####################################### Hierarchical document clustering #############################
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(8, 5)) # set size
ax = dendrogram(linkage_matrix, orientation="right",labels=books_titles,show_leaf_counts=True);

plt.tick_params(
    axis= 'y',          # changes apply to the x-axis
    which='off',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

################################### Visualization on 2 Axis #####################################
#set up colors per clusters using a dict
#cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a'}
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4:'#000000',5:'#0fD3A0'}

#set up cluster names using a dict
cluster_names = {0: 'Jews, Herod, Josephus, Jerusalem',
                 1: 'Tribunes, Consuls, Commons, Chap, Dictator',
                 2: 'Lib, Emperor, Chap, Justinian, Decline',
                 3: 'Athenians, Peloponnesian, Fay',
                 4: 'Nero, Tacitus, Otho, Germanicus',
                 5: 'Nero, fds, Otho, Germanicus'
                 }

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=books_titles))

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)

plt.show() #show the plot

#uncomment the below to save the plot if need be
#plt.savefig('clusters_small_noaxes.png', dpi=200)

###################################### Word Cloud of Clusters ###################################
cluster_words = []
for word in order_centroids[3]:
    cluster_words.append(terms[word])

text = " "
for x in cluster_words:
    text = text + x + " "

wordcloud = WordCloud(width = 1400, height = 700,#
                      background_color ='white',
                      stopwords = list_stopwords,
                      min_font_size = 10).generate(text)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)