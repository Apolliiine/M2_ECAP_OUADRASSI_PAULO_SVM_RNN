# **Projet Machine Learning**

Notre étude porte sur un jeux de données liées aux campagnes de marketing direct d'une institution bancaire portugaise. Nous allons réaliser un projet de machine learning en utilisant la classification (SVM et réseaux de neurones) afin de prédire combien de personnes vont ouvrir un compte bancaire avec l'institution bancaire portugais.

Cette variable, à prédire est représenté par une variable binaire qui prendra la valeur de 1 lorsque une personne décide d'ouvrir un compte bancaire avec l'institution bancaire portugais, sinon elle prendra la valeur de 0.

Nous allons commencer par analyser et nettoyer notre jeu de données (analyse des variables, identification valeurs manquantes, outliers), Ensuite, nous allons réaliser des classifications. Dans une premier partie nous allons réaliser une classification multiclass en utilisant deux approches OVR(one versus rest) et OVO(one versus one). Pour terminer, nous allons réaliser une classification avec keras.

Source des données :https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset?select=train.csv

## **I. Analyse du jeu de données**

## Visualisation de notre variable à expliquer

![Pie chart y](Images/piechart_y.png "Répârtition de la variable à expliquer")

![Pie chart y](Images/barplot_y.png "Répârtition de la variable à expliquer")

D'après ces représentations graphiques, nous constatons qu'environ 88% d'individus n'ont pas ouvert un compte bancaire avec l'institution bancaire portugais.

## Identification et correction des valeurs manquantes
Nous allons vérifier qu'il n'y ait pas de valeurs manquantes et procéder au traitement de celles-ci dans le cas où il y'en aurait (suppression, imputation, etc.)

Nous constatons que trois de nos variables ont de valeurs atypiques(default, housing, loan). Nous avons choisi de corriger ces valeurs en utilisant la fonction fillna, en remplaçant les valeurs manquantes par le mode de la série.

# Lien entre variables 


> Nous allons utiliser la corrélation de Spearman afin d'étudier les liens entre les variables explicatives.

![Pie chart y](Images/corr_mat.png "Répartition de la variable à expliquer")

La matrice de confusion obtenue ci-dessus permet d'évaluer les performances d'un classement en comparant les valeurs prédites par le modèle aux valeurs réelles.

Voici les prédictions de notre modèle :

12473 personnes ne vont pas ouvrir un compte bancaire avec l'institution bancaire portugais et a contrario 294 personnes vont ouvrir un compte bancaire avec cette institution, ces prédictions correspondent à la réalité.
Dans cette matrice de prédiction, nous avons également des faux positives et faux négatives. En effet, le modèle a prédit que 1375 individus allaient ouvrir un compte bancaire avec l'institution portugais, or ce n'était pas le cas. Il a également prédit que 181 individus n'allaient pas ouvrir un compte bancaire avec cette institution, alors que dans la réalité, ils ont ouvert.

# Classification avec Keras

![Pie chart y](Images/keras1.png "Répartition de la variable à expliquer")

On a 4801 paramètres dans ce réseau de neurones. La couche d'entrée contient 100 neurones, chacun de ses neurones est associé à nos feautures. On rajoute pour chaque neurone un terme de biais (ceci nous permet d'obtenir 4700 paramètres). La couche de sortie contient 1 biais plus les 100 neurones de la couche précédente. Ceci nous permet de retrouver les 4801 paramètres.



![Pie chart y](Images/keras2.png "Répartition de la variable à expliquer")

On a 25001 paramètres dans ce réseau de neurones. La couche d'entrée contient 100 neurones, chacun de ses neurones est associé à nos features. On rajoute pour chaque neurone un terme de biais (ceci nous permet d'obtenir 4700 paramètres). Ce modèle est constitué de deux couches cachées qui contiennent également 100 neurones multipliés par les 100 neurones en entrée. La couche de sortie contient 1 biais plus les 100 neurones de la couche précédente. Ceci nous permet de retrouver les 25001 paramètres.

![Pie chart y](Images/lc3.png "Répârtition de la variable à expliquer")
Sur le graphique ci-dessus, nous pouvons voir la fonction de perte représentée en bleu et l'accuracy.
Contrairemetn au auxtres modèles, la fonction de perte connait des augmentations et des baisses en fonction du nombe d'epochs. Elle ne diminue pas continuellement. L'accuracy est plutôt constante quelque soit le nombre d'epochs, malgré quelque légère hausses et baisses.


Pour terminer, nous avons utilisé la fonction accuracy afin de prédir l'efficacité de modèle. Nos modèles ont des prévisions efficace à 88.35%.

# **Conclusion**
L'objectif de notre projet était de prédire combien de personnes vont ouvrir un compte bancaire avec l'institution bancaire portugais.

Nous avons commencé par réaliser une analyser et traiter les données en réalisant des statistiques descriptives, vérification de la corrélation entre nos variables explicatives et nettoyer nos données. Nous avons fait le choix de ne pas supprimer les individus atypiques parce que ça réduisait notre base de données de façon significative.

Ensuite, nous avons réalisé des classifications afin de pouvoir prédire notre variable.

Nous avons commencé par réaliser une classification multiclass, cette dernière nous a permis d'obtenir la conclusion suivante : 12473 personnes ne vont pas ouvrir un compte bancaire avec l'institution bancaire portugais et a contrario 294 personnes vont ouvrir un compte bancaire avec cette institution, ces prédictions correspondent à la réalité. Dans cette matrice de prédiction, nous avons également des faux positives et faux négatives. En effet, le modèle a prédit que 1375 individus allaient ouvrir un compte bancaire avec l'institution portugais, or ce n'était pas le cas. Il a également prédit que 181 individus n'allaient pas ouvrir un compte bancaire avec cette institution, alors que dans la réalité, ils ont ouvert un compte bancaire.

Pour terminer, nous avons réalisé une classification avec keras. Nous avons réalisé trois modèles, dont un, il a deux couches cachées. Ces modèles ont respectivement 4801,25001 et 481 paramètres. Nous avons ensuite utilisé la fonction accuracy afin de prédire l'efficacité de modèles. Nous constatons que les trois modèles ont des prévisions efficaces à 88.35%.
