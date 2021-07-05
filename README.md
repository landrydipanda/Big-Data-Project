# Projet_BigData_LandryDipanda_SidiToure
Ce projet s'inscrit dans le cadre du module "Big Data" dispensé en cinquième de notre formation de cycle ingénieur.

Le but du projet c'est de construire un classifieur de documents à l'aide de l'algotihme de clustering k-means++ en partant d'un lac de données etiquités de documents.

- Jeux de données : https://www.kaggle.com/nzalake52/new-york-times-articles

Compte rendu : 
- Big Data / machine learning 
    - Preprocessing / nettoyage du corpus
    - Feature engeneering
    - Feature selection
    - Vectorisation 
    - Clustering en utilisant K-means++ dans SPARK
    - Metrique d'avaluation des clusters : purity metric, silhouete score
    - Optimisation des resultats des métriques 
- Data Vizuation 
    - Association des métadonnées pour chaque document
    - Preparation des exports CSV pour Data Viz sur tensorflow projector : https://projector.tensorflow.org/
    - Convergence des clusters
- Remarques : chaque cluster entrainé répresentera une classe du lac de données. La precision des classes est evalué à l'aide de la "purity metrics"
- Résultats 
    - Purity metric : 0.9/1 
- Documentation
    - Article scientifique utilisé : https://www.researchgate.net/publication/236124137_Information-theoretic_Term_Weighting_Schemes_for_Document_Clustering
    - Rapport théorique du projet : https://drive.google.com/file/d/1OIW3TIMq9uRz1FFarJ_ukKfHDp8RKipa/view?usp=sharing
