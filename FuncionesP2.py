# -*- coding: utf-8 -*-
"""
@author: Carlos
"""

import time

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, DBSCAN, Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics,preprocessing
from scipy.cluster import hierarchy
from math import floor
import seaborn as sns

"""
    Clustering
"""

def kmeans(X, X_norm, n_cl, n_in, SEED, usadas, caso):
    print("----- Ejecutando kmeans para {} clusters".format(n_cl),end="")
    k_means = KMeans(init="k-means++", n_clusters=n_cl, n_init=n_in, random_state=SEED)
    
    t = time.time()
    cluster_predict = k_means.fit_predict(X_norm)
    tiempo = time.time() - t
    print(": {:.2f} segundos".format(tiempo), end="\n")
    
    metrics_CH = metrics.calinski_harabasz_score(X_norm, cluster_predict)
    print("Calinski-Harabaz Index: {:.3f}".format(metrics_CH), end="\n")
    
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
    metric_SC = metrics.silhouette_score(X_norm, cluster_predict, metric="euclidean", sample_size=floor(muestra_silhoutte*len(X)), random_state=SEED)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC), end="\n")
    
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=["cluster"])
    print("Tamaño de cada cluster:")
    size = clusters["cluster"].value_counts()
    for num,i in size.iteritems():
        print("%s: %5d (%5.2f%%)" % (num,i,100*i/len(clusters)))
    centers = pd.DataFrame(k_means.cluster_centers_,columns=list(X))
    
    visualizar(X, usadas, centers, clusters, caso, "kmeans")
    
def kmeans_compare(X, X_norm, n_cl_max, n_in, SEED, usadas, caso):
    print("----- Ejecutando kmeans para comparar en funcion del numero de clusters", end="\n")
    f = open('./{}/kmeans/comparacion.txt'.format(caso), 'w')
    for nc in range(2,n_cl_max):
        f.write("----- Ejecutando kmeans para {} clusters".format(nc))
        k_means = KMeans(init="k-means++", n_clusters=nc, n_init=n_in, random_state=SEED)
        
        t = time.time()
        cluster_predict = k_means.fit_predict(X_norm)
        tiempo = time.time() - t
        f.write(": {:.2f} segundos\n".format(tiempo))
        
        metrics_CH = metrics.calinski_harabasz_score(X_norm, cluster_predict)
        f.write("Calinski-Harabaz Index: {:.3f}\n".format(metrics_CH))
        
        muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
        metric_SC = metrics.silhouette_score(X_norm, cluster_predict, metric="euclidean", sample_size=floor(muestra_silhoutte*len(X)), random_state=SEED)
        f.write("Silhouette Coefficient: {:.5f}\n".format(metric_SC))
        
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=["cluster"])
        f.write("Tamaño de cada cluster:\n")
        size=clusters["cluster"].value_counts()
        for num,i in size.iteritems():
            f.write("%s: %5d (%5.2f%%)\n" % (num,i,100*i/len(clusters)))
            
def meanshift(X, X_norm, SEED, usadas, caso):
    print('----- Ejecutando MeanShift',end='')
    
    bwth = estimate_bandwidth(X_norm, random_state = SEED, n_samples = 500, quantile = 0.2)
    ms = MeanShift(bandwidth = bwth, bin_seeding = True)
    
    t = time.time()   
    ms.fit(X_norm)	
    tiempo = time.time() - t
    
    labels = ms.labels_
    print(": {:.2f} segundos".format(tiempo), end='\n')
    
    metrics_CH = metrics.calinski_harabasz_score(X_norm, labels)
    print("Calinski-Harabaz Index: {:.3f}".format(metrics_CH), end='\n')
    
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
    metric_SC = metrics.silhouette_score(X_norm, labels, metric="euclidean", sample_size=floor(muestra_silhoutte*len(X)), random_state=SEED)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC), end="\n")
    
    clusters = pd.DataFrame(labels,index=X.index,columns=['cluster'])
    
    print("Tamaño de cada cluster:")
    size = clusters["cluster"].value_counts()
    for num,i in size.iteritems():
        print("%s: %5d (%5.2f%%)" % (num,i,100*i/len(clusters)))
    centers = pd.DataFrame(ms.cluster_centers_,columns=list(X))
    
    visualizar(X, usadas, centers, clusters, caso, "meanshift")
    
def dbscan(X, X_norm, eps, SEED, usadas, caso):
    print('----- Ejecutando dbscan', end='')	
    
    e = eps
    dbs = DBSCAN(eps = e)
        
    t = time.time()
    dbs.fit(X_norm)
    tiempo = time.time() - t   
    
    print(": {:.2f} segundos".format(tiempo), end='\n')
    
    labels = dbs.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    metrics_CH = metrics.calinski_harabasz_score(X_norm, labels)
    print("Calinski-Harabaz Index: {:.3f}".format(metrics_CH), end='\n')
    
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
    metric_SC = metrics.silhouette_score(X_norm, labels, metric="euclidean", sample_size=floor(muestra_silhoutte*len(X)), random_state=SEED)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC), end="\n")
    
    print('Epsilon: {}'.format(e))
    print('Número estimado de clusters: {}'.format(n_clusters_))
    print('Número estimado de puntos ruidosos: {}'.format(n_noise_))
    
    clusters = pd.DataFrame(labels,index=X.index,columns=['cluster'])
    print("Tamaño de cada cluster:")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        
    X_DBSCAN = pd.concat([X_norm,clusters],axis=1)
    X_DBSCAN = X_DBSCAN[X_DBSCAN.cluster != -1]
    cluster_centers = X_DBSCAN.groupby('cluster').mean()
	
    centers = pd.DataFrame(cluster_centers,columns=list(X))
    visualizar(X, usadas, centers, clusters, caso, 'dbscan')
    
def dbscan_compare(X, X_norm, eps_max, SEED, usadas, caso):
    e = 0.15
    f = open('./{}/dbscan/comparacion.txt'.format(caso), 'w')
    while e < eps_max:
        f.write("----- Ejecutando dbscan para epsilon {}".format(e))
        dbs = DBSCAN(eps = e)
            
        t = time.time()
        dbs.fit(X_norm)
        tiempo = time.time() - t   
        
        f.write(": {:.2f} segundos\n".format(tiempo))
        
        labels = dbs.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        
        metrics_CH = metrics.calinski_harabasz_score(X_norm, labels)
        f.write("Calinski-Harabaz Index: {:.3f}\n".format(metrics_CH))
        
        muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
        metric_SC = metrics.silhouette_score(X_norm, labels, metric="euclidean", sample_size=floor(muestra_silhoutte*len(X)), random_state=SEED)
        f.write("Silhouette Coefficient: {:.5f}\n".format(metric_SC))
        
        f.write('Epsilon: {}\n'.format(e))
        f.write('Número estimado de clusters: {}\n'.format(n_clusters_))
        f.write('Número estimado de puntos ruidosos: {}\n'.format(n_noise_))
        
        clusters = pd.DataFrame(labels,index=X.index,columns=['cluster'])
        f.write("Tamaño de cada cluster:\n")
        size=clusters['cluster'].value_counts()
        for num,i in size.iteritems():
            f.write('%s: %5d (%5.2f%%)\n' % (num,i,100*i/len(clusters)))
        e += 0.05
        
    
def birch(X, X_norm, n_cl, thld, SEED, usadas, caso):
    print("----- Ejecutando birch para {} clusters".format(n_cl), end="")
    b = Birch(n_clusters = n_cl, threshold = thld)        
    
    t = time.time()
    b.fit(X_norm)
    tiempo = time.time() - t
    print(": {:.2f} segundos, ".format(tiempo), end="")
    
    labels = b.labels_
        
    metrics_CH = metrics.calinski_harabasz_score(X_norm, labels)
    print("Calinski-Harabaz Index: {:.3f}".format(metrics_CH), end="\n")
    
    muestra_silhoutte = 0.2 if (len(X) > 10000) else 1.0
    metric_SC = metrics.silhouette_score(X_norm, labels, metric="euclidean", sample_size=floor(muestra_silhoutte*len(X)), random_state=SEED)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC), end="\n")
    
    clusters = pd.DataFrame(labels,index=X.index,columns=["cluster"])
    size=clusters["cluster"].value_counts()
    
    print("Tamaño de cada cluster:")
    for num,i in size.iteritems():
        print("%s: %5d (%5.2f%%)" % (num,i,100*i/len(clusters)))
    
    X_birch = pd.concat([X_norm,clusters],axis=1)
    cluster_centers = X_birch.groupby("cluster").mean()
	
    centers = pd.DataFrame(cluster_centers,columns=list(X))
    visualizar(X, usadas, centers, clusters, caso, "birch")
    
def jerarquico(X, n_cl, SEED, usadas, caso):
    print('----- Ejecutando Jerarquico',end='')
    
    if len(X)>1000:
        X = X.sample(1000, random_state = SEED)
    X_norm = preprocessing.normalize(X)
    
    ward = AgglomerativeClustering(n_clusters=n_cl, linkage='ward')
    name, algorithm = ('Ward', ward)
    cluster_predict = {}
    k = {}
    
    print(name,end='')
    t = time.time()
    cluster_predict[name] = ward.fit_predict(X_norm)
    tiempo = time.time() - t
    k[name] = len(set(cluster_predict[name]))
    print(": k: {:3.0f}, ".format(k[name]),end='')
    print("{:6.2f} segundos".format(tiempo))
    
    clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
    X_cluster = pd.concat([X, clusters], axis=1)
    
    min_size = 10
    X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
    k_filtrado = len(set(X_filtrado['cluster']))
    print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k['Ward'],k_filtrado,min_size,len(X),len(X_filtrado)))
    X_filtrado = X_filtrado.drop(['cluster'], axis=1)
    print("")

    X_filtrado = X.copy()
    X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')
    
    linkage_array = hierarchy.ward(X_filtrado_normal)
    plt.figure(1)
    plt.clf()
    dendro = hierarchy.dendrogram(linkage_array,orientation='left')
    plt.savefig('./{}/{}/dendograma.png'.format(caso, "jerarquico"), dpi=200)
    plt.clf()
    #lo pongo en horizontal para compararlo con el generado por seaborn
    #puedo usar, por ejemplo, "p=10,truncate_mode='lastp'" para cortar el dendrograma en 10 hojas
    
    
    #Ahora lo saco usando seaborn (que a su vez usa scipy) para incluir un heatmap
    plt.figure(2)
    X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)
    g=sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)
    plt.savefig('./{}/{}/dendogramaHeatmap.png'.format(caso, "jerarquico"), dpi=200)
    plt.clf()
    #g.savefig(algorithm + ".pdf")
    
"""
    Gráficas
"""

def visualizar(X, usadas, centers, clusters, caso, alg):
    print("---------- Preparando el heatmap...")
    sns.set()
    centers_desnormal = centers.copy()
    
	#se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())
	
    axhm = plt.axes()
    hm = sns.heatmap(centers, cmap='YlGnBu', annot=centers_desnormal, annot_kws={"fontsize":20}, fmt='.3f', ax=axhm)
    axhm.set_title('Heatmap')    
    plt.xticks(rotation=30)
    hm.set_ylim(len(centers),0)
    hm.figure.set_size_inches(15,15)
    hm.figure.savefig('./{}/{}/heatmap.png'.format(caso, alg), dpi=200)
    print("")
    plt.clf()
    
    
    print("---------- Preparando el scatter matrix...")
    sns.set()
    X_alg = pd.concat([X, clusters], axis=1)
    variables = list(X_alg)
    variables.remove('cluster')
	
    sns_plot = sns.pairplot(X_alg, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist")
    sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03)
    sns_plot.savefig('./{}/{}/scatter_matrix.png'.format(caso, alg), dpi=200)
    print("")
    plt.clf()
    
    
    print("---------- Preparando el box plot...")
    sns.set()
    size=clusters['cluster'].value_counts()
    k = len(size)
    n_var = len(usadas)
    fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(15,10))
    colors = sns.color_palette(palette=None, n_colors=k, desat=None)
	
    rango = []
    for j in range(n_var):
        rango.append([X_alg[usadas[j]].min(),X_alg[usadas[j]].max()])
	
    for i in range(k):
        dat_filt = X_alg.loc[X_alg['cluster']==i]
        for j in range(n_var):
            ay = sns.boxplot(x=dat_filt[usadas[j]], color=colors[i], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])
            if (i==k-1):
                ay.set_xlabel(usadas[j])            
            else:
                ay.set_xlabel("")
            #ay.autoscale(enable=True)
            ay.set_xlim(rango[j][0],rango[j][1])
    
    fig.savefig('./{}/{}/boxplot.png'.format(caso, alg), bbox_inches='tight', dpi=200)
    plt.clf()
    print("")