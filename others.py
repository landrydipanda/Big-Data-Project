

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter 
from nltk import word_tokenize #tokenization

from nltk.stem import WordNetLemmatizer

#Permet de verifier si un chiffre est contenue dans une chaine de caractère donnée
def number_in_str(x):
    rt=False
    nbs=['0','1','2','3','4','5','6','7','8','9']
    for c in nbs:
        if (c in x):
            rt=True
    return rt

#Extration des indexes des lignes qui sont indicatrices de categorie 
def extract_id_categories(lines):
    id_categories=[]
    for i in range (len(lines)):
        if(len(lines[i])>10):
            if lines[i][:10] =='URL: http:':
                id_categories.append(i)
    return id_categories

#lemmatisation and pos tagging 
def lemmatisation_and_pos_tagging(doc):#doc : article de type 'str'
    #Definition de la classe pour la lemmmatization des articles
    lemmatizer = WordNetLemmatizer() 
    #Specification des formes gramaticales(verbe, adverbe, preposition) que nous ne souhaiterons pas garder
    remove_pos_tagging=set(["CD","CC","DT","EX","IN","LS","MD","PRP","PRP$","TO","UH","WRB","VB","VBD","VBG","VBN","VBP","VBZ","WP$","WP","WDT","JJ","JJR","JJS","RB","RBR","RBS"]) #

    doc_='' #new article with appliying lemnisation and pos tagging
    wds=word_tokenize(doc)#list: wds in article (tokenization)
    #wds=nltk.pos_tag(wds)
    #wds=[lemmatizer.lemmatize(wd[0]) for wd in wds if (wd[1] not in remove_pos_tagging)] #Pos tagging ?
    wds=[lemmatizer.lemmatize(wd) for wd in wds]
    wds=[wd for wd in wds if len(wd)>3]
    doc_=doc_+" ".join(wd for wd in wds)
        
    return doc_

#Extraction des mots discriminant dans chaque categorie : utilisation de la metrique tf-idf
def features_extraction(documents,labels_categories):   
    #print(documents)
    #Regrouper tous les articles d'une même collection dans un seul string 
    categories={}
    for i in range (len(documents)):
        if(labels_categories[i] not in categories.keys()):
            categories[labels_categories[i]]=documents[i]
        else:
            categories[labels_categories[i]]=categories[labels_categories[i]]+' '+documents[i]
    #print(categories)
    cat=[x for x in categories.keys()]
    tf_idf_categories=[x for x in categories.values()]
    #print(tf_idf_categories)
    vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,2),use_idf=True) #2000 :0.8854732933325533% avec Pos tagging
    X = vectorizer.fit_transform(tf_idf_categories)
    X=X.toarray()#inclure la presence des zeros dans les vecteurs

    #Recuperation de tous les features
    all_features=vectorizer.get_feature_names()

    #Extraction des features pertinents par categories
    features_count={}
    for x in range(len(cat)):
        features_count[cat[x]]=[]
        for i in range(len(all_features)):
            features_count[cat[x]].append([all_features[i],X[x][i]])#[feature,tf-idf]
    for k,v in features_count.items():
        features_count[k]=sorted(v,reverse=True,key=lambda t: t[1])  #tri par ordre de frequence decroissant
        features_count[k]=features_count[k][:800] #couper
        features_count[k]=[x[0] for x in features_count[k]] #prendre que les mots

    best_features=[]
    for k,v in features_count.items():
        list_wds=[]
        for k_oth in features_count.keys():
            if k_oth!=k:
                list_wds=list_wds+features_count[k_oth]

        i=0 ;  mx=200 ; ls=[] #on prend  150 features discriminants par categories
        while(len(ls)<mx and i<len(v)):
            if(v[i] not in set(list_wds)):
                ls.append(v[i])
            i=i+1
        print(k + "  "+str(len(ls)))
        print(ls)
        best_features=best_features+ls




    """
    #Tri , selection 
    for k,v in features_count.items():
        features_count[k]=sorted(v,reverse=True,key=lambda t: t[1])  #tri par ordre de frequence decroissant
        for i in range(200):
            best_features.append(features_count[k][i][0])
    """
    print(len(best_features))
    #best_features=set(best_features)
    #print(len(best_features))
    return best_features

