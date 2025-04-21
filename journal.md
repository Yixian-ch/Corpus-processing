# Projet
## Étude sur CoNLL 2003
1. Tâche : Reconnaissance des entités nommées indépendantes des lanuges
2. Type de données : Les données sont des corpus classés en anglais et en allemand dont chacun contient 4 types de données : training set, test set, developpement file et raw file
   - Les corpus anglais sont collectés par Reuters Corpora dont les formats possibles sont XML, SGML... Ils sont donc des documents semi-structurés selon la définiton du cours.
   - Pour les corpus allemands, ils sont collectés par ECI Multilingual Text Corpus dont la plus part de corpus sont en SGML, ils sont donc de documents semi-structurés.
4. Besoin : On a des corpus en entrée et les sorties souhaitées sont les entités nommées. Donc nos besoins être de reconnaître les entités nommées.
5. Type de modèles : Modèles neuronaux d'apprentissage profond comme Bert.
6. Lanuge : ils sont de corpus multilingues : anglais, allemand.

## Étapes du projet
1. constituer un corpus avec webcrawler
2. évaluer la qualité du corpus avec les différents outils
3. stocker les données dans un format accessible
4. choisir un modèle à évaluer sur le corpus
5. fine tuning
   
### Choix de la tâche
1. son besoin en ressources
2. quel modèle on utilise
3. les langues que l'on va traiter
4. monolingue/multilingue
5. corpus parallèle?
6. genre des textes (résumé d'article de journaux vs résumé de décision juridique)

### critères de définition d'un corpus
1. Taille
2. annotations
3. statut de la documentation (guidelines, publication scientifique)
4. stratégie d'échantillonnage et origine des textes (genre)
5. licence de droits d'utilisation
Tout ça se trouve dans la carte des corpus
où trouver les corpus : `https://www.ldc.upenn.edu/`, `https://www.elra.info/`, Kaggle
Pour la tâche Summarization : `cnn_dailymail`
#### Constituer un corpus
- l'analyse d'un corpus pré-existant
- constitution d'un corpus similaire (forme, méthode de récupération, combien de lignes, de colonnes et de classes). L'évaluation de la qualité de données: quartet d'anscombe, test de correlation, p_value, loi de zipf, heaps. Nettoyage de donnes. (outliers, deduped)
- bon code (docstring, lisibilité, la conventions telles que PEP8, réutilisabilité, maintenabilité)
- visualisation de données
- évaluation du corpus

### Sur les données
1. Quantité des données dépend de la tâche et du modèle utilisé.
2. Équilibre de données: la variété de données
3. Trois façons de récupérer les données : base de données, API, crawler (savoir lire robots.txt)

   
### Structure de projet
PROJECT/
├── data/ 
│ ├── raw/ 
│ └── clean/ 
│ 
├── figures/ <- figures used in place of a "results"folder. 
├── src/ 
│ ├── process/ <- scripts to maniuplate data between raw, cleaned final stages. 
│ └── plot/ <- intermediate plotting. 
│ 
├── bin 
│ ├── model1/ <- various experimental models. 
│ ├── model2/ 
│ └── model3/
│ 
├── LICENSE 
├── Makefile 
└── readme.md



