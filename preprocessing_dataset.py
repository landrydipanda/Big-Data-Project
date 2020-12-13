
# -*- coding: utf-8 -*-
import re #Expressions regulière
from statistics import mean,stdev,variance #stdev : ecart type 

#Verifie si un chiffre est contenue dans une chaine de caractère donnée
def number_in_str(x):
    rt=False
    nbs=['0','1','2','3','4','5','6','7','8','9']
    for c in nbs:
        if (c in x):
            rt=True
    return rt

#Affichage des informations concernant la structure du Dataset
if __name__=="__main__":   
    #Ouverture du dataset	
    file=open("nytimes_news_articles.txt","r",encoding='utf-8')
    #recuperation des lignes dans le fichier: une ligne est consitué d'une ou de plusieurs phrases
    lines=file.readlines()#une ligne est une chaine de caractère

    #preprocessing du texte
    #remplacement des caractères '\n' par le caractère ''
    lines=[re.sub(r'\n',r'',line) for line in lines]#chaque element de la liste est un chaine de caractère qui comporte un ensemble de phrases
    #Retirer les chaines de caractère de la liste qui ne contiennent aucun mot 
    lines=[line for line in lines if len(line)>0]

    #Recuperation des indices des lignes qui sont des indicateurs de categories : 8 888 articles au total , nb categories
    #chaine de caractère = un element de la liste lines[]
    id_categories=[]#len(id_categories)
    for i in range (len(lines)):
        if(len(lines[i])>10):
            if lines[i][:10] =='URL: http:':#une ligne qui commence par 'URL: http:' est un indicateur de categorie
                id_categories.append(i)	

    #Recuperation des articles par categorie
    categories={} #declaration d'un dictionnaire vide 
    """
    clé : categorie ; valeur : [[#texte],[#nb mots]]
    Un article est caracteisé par un string (chaine de caractère)
    """
    nb_articles=0#compte le nombre d'articles dans le corpus
    list_nb_wds=[] #liste: contient le nombre de mots de chaque article dans le corpus 
    for i in range(len(id_categories)): #Parcours des indices des lignes qui sont des indicateurs de categorie
        str_=lines[id_categories[i]].split('/')
        category=str_[-2]
        if (number_in_str(str_[-3])==False):
            category=str_[-3]+'/'+category
        #category=category.split('/')[0]#
        if i<len(id_categories)-1:
            lst=id_categories[i+1]
        else:
            lst=len(lines)
        #Recuperation du texte de l'article: 
        text=""
        for j in range(id_categories[i]+1,lst):
            text=text+" "+lines[j]
        #Comptage du nombre de mots de l'article
        nb_wds=len(text.split()) ; list_nb_wds.append(nb_wds)
        #Sauvegarde de l'article dans la catagorie à l'aide du dictionnaire 
        if category not in categories.keys():
            categories[category]=[[text],[nb_wds]]
            nb_articles=nb_articles+1
        else:
            categories[category][0].append(text)
            categories[category][1].append(nb_wds)
            nb_articles=nb_articles+1
        
    print('Nombre de categories : '+str(len(categories)))
    print("Nombre d'articles dans le corpus : "+str(nb_articles))
    print("La moyenne (en terme de mots ) des articles globale : "+str(mean(list_nb_wds)))
    print("\nAffichage du nombre d'articles par categorie\n")
    print("Categorie"+'\t'+"Nb articles dans la categorie\t"+"Moyenne des articles (en terme de mots) dans cette categorie")
    for key in sorted(categories):
        print(key +' \t '+str(len(categories[key][0]))+" \t "+str(mean(categories[key][1]))+"\t")