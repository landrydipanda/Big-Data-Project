
# -*- coding: utf-8 -*-
import re
#from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter 
from statistics import mean

import nltk
#nltk.download()
from nltk import word_tokenize,sent_tokenize

from others import number_in_str,extract_id_categories,lemmatisation_and_pos_tagging,features_extraction


#Ouverture du fichier '.txt' qui contient le corpus en mode lecture	
file=open("nytimes_news_articles.txt","r",encoding='utf-8')
lines=file.readlines()#Recuperation des lignes du fichier texte (corpus)
    
#...................Preprocessing du texte
#remplacement des caractères '\n' par le caractère ''
lines=[re.sub(r'\n',r'',line) for line in lines]#chaque element de la liste est un chaine de caractère qui comporte un ensemble de phrases
#Retirer les chaines de caractère de la liste qui ne contiennent aucun mot 
lines=[line for line in lines if len(line)>0]

#.................. Extration des indexes des lignes qui sont indicatrices de categorie : 8 888 categories
id_categories=extract_id_categories(lines)

#...................Precison des categories que l'on souhaite traiter: 
topic_=['dining','fashion','sports basketball','technology','world middleeast']#,'business','science'] 
nb_clus=len(topic_)

#les listes ci-dessous sont de même taille et permettent de recuperer les informations relative à chaque article
labels_categories=[]
documents=[]
vectors=[]
labels_clusters=[]

"""
Chaque article est indéxé par un indice "id" qui lui permet de s'identifier comme suit : 
#labels_categories[id]: categorie auquels appartient l'article
#documents[id] : le texte contenue dans l'article
#vectors[id] : le vecteur de l'article
#labels_clusters : l'identifiant du cluster auquelle appartient l'article

"""

#Remplissage des listes : labels_categories , documents
for i in range(len(id_categories)): 
	#Recuperation de la categorie : sub_topic
	top=lines[id_categories[i]].split('/')
	sub_topic=top[-2]
	if (number_in_str(top[-3])==False):
		sub_topic=top[-3]+' '+sub_topic
	if sub_topic in topic_:#On se rassure que la categorie fait partir de celle qui nous interesse
		if i<len(id_categories)-1:
			lst=id_categories[i+1]
		else:
			lst=len(lines)
		#Recuperation du texte de l'article:
		text=""
		for j in range(id_categories[i]+1,lst):
			text=text+" "+lines[j]
		
		labels_categories.append(sub_topic)
		documents.append(text)

#...................Others preprocessing
#Remplacement des chiffres (contenu dans les texte) par un vide (caractère : "")
print('remove digits from corpus')
documents=[re.sub("\d+", "", doc) for doc in documents]
print('lemnization and/or pos tagging')
#Lemmatisation et Pos taggging sur les articles
documents=[lemmatisation_and_pos_tagging(doc) for doc in documents]

#...................Vectorisation  des articles : 

#print('features extraction using frequence :')
#vectorizer = TfidfVectorizer(stop_words='english',max_features=1000,ngram_range=(1,2),use_idf=True)

print('features extraction using tf-idf :')
features_using_tf_idf=features_extraction(documents=documents,labels_categories=labels_categories)

print("vectorisation des articles à partir de la metrique : tf-idf")
vectorizer = TfidfVectorizer(vocabulary=features_using_tf_idf,stop_words='english',ngram_range=(1,2),use_idf=True) #2000 :0.8854732933325533% avec Pos tagging
X = vectorizer.fit_transform(documents)
X=X.toarray()
vectors=list(X)#mise à jour la matrice des vecteurs des données

#...................Clusetring using K-means algorithm:
print('clusetring using k-means algorithm')

kmeans = KMeans(n_clusters=5,random_state=0,max_iter=1000,n_jobs=-1).fit(X) #nb_clus: 25 , precision : 0.98 , 05 clusters
#print(kmeans.cluster_centers_)

#Recuperation des labels des clusters
labels_clusters=list(kmeans.labels_)

#................Clustering evaluation usign purity metric
print("clusetring evaluation using purity metric")

categories_in_cluster={}
cluster_purity={}
mean_cluster_purity=[]

"""
labels_clusters[id] : cluster de l'article indexé par "id"
labels_categories : categorie de l'article indéxé par 'id'
"""
#Regroupement de l'ensemble des categories présents danc chaque cluster
for i in range(len(labels_clusters)):
	if (labels_clusters[i] not in categories_in_cluster.keys()):
		categories_in_cluster[labels_clusters[i]]=[labels_categories[i]]
	else:
		categories_in_cluster[labels_clusters[i]].append(labels_categories[i])



#Calcul des puretes de chaque cluster et affichage des categories predominantes
for cluster,categories in categories_in_cluster.items():
	occ_categories=Counter(categories)#Donne le nombre d'occurence de chaque categorie present dans chaque cluster
	occ_categories=dict(sorted(occ_categories.items(),reverse=True,key=lambda t: t[1]) )#Tri du dictionnaire 'occ_categories'
	predominant_category=list(occ_categories.keys())[0]#Recuperation de la categorie predominante dans le cluster

	#Calcul de la puréte du cluster
	cluster_purity[cluster]=[predominant_category,occ_categories[predominant_category]/len(categories)]

	mean_cluster_purity.append(occ_categories[predominant_category]/len(categories))
	print('la categorie "'+predominant_category+'" predomine dans le cluster N° '+str(cluster) + " avec un purete de : "+str(occ_categories[predominant_category]/len(categories)))

#Calcul de la purete globale des cluster == moyenne des puretes de chaque cluster
total_purity=mean(mean_cluster_purity)
print("Moyenne des purete des clusters (purété global) : "+str(total_purity))


#Mise en visuel tensorflow projector
#sauvegarde csv pour data viz en locale -> pb de fluidite et rapidité sur le serveur
with open("projections/metadata.tsv", 'w+',encoding="utf-8") as file_metadata , open("projections/vectors.tsv", 'w+',encoding="utf-8") as file_vectors:
	file_metadata.write("Categorie"+"\t"+"Cluster"+"\t"+"Categorie Predominante"+"\t"+"pureté"+"\t"+"extrait"+"\n")
	for i in range(len(vectors)):
		vec=vectors[i] #recuperation des coordonnées de l'element en cours
		coords=str(vec[0])
		for k in range(1,len(vec)):
			coords=coords+"\t"+str(vec[k])
		file_vectors.write(str(coords)+"\n")
		#metadata :
		file_metadata.write(str(labels_categories[i])+"\t"+str(labels_clusters[i])+"\t"+cluster_purity[labels_clusters[i]][0]+"\t"+str(cluster_purity[labels_clusters[i]][1])+"\n") #indice -> 0: mot ; 1: frequence   




"""
Sum_of_squared_distances = []
K = range(nb_clus,nb_clus+50)
for k in K:
	km = KMeans(n_clusters=k,random_state=1000,n_jobs=-1)
	km = km.fit(X)
	Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
"""