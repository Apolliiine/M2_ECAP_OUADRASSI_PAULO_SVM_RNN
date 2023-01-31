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




