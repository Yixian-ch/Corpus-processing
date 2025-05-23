# Projet TAL 2024-2025
## Définition de besoins
- besoins : à partir des desctiptions textuelles pour la mission handicap développer un modèle de NLG (natural language genaration) .
- sujet : Génération automatique des contenus du sujet : mission handicap.
- tâche : NLG.
- données : des données non structurées collectées par les scripts de webcrawler.
- `https://www.sorbonne-nouvelle.fr/qu-est-ce-que-le-handicap--402898.kjsp?RH=1474288330206`
- `https://www.inalco.fr/etudiant-en-situation-de-handicap`
- `https://www.nanterre.fr/annuaires/catalogue-des-demarches/detail/handicaps-au-quotidien`
- Après avoir consulté les indications données dans les pages robots de chaque universités, les 3 pages que j'ai choisies ne sont pas dans la liste non autorisée.
## TP2
Pour collecter les données depuis l'Internet, j'ai développé un script de web scraping, puisque je sais exactement à partir de quelle page je veux constituer mon corpus. Pour se faire, j'ai d'abord consulté les fichiers robots.txt de chaque université, et ai comfirfé que les pages que je vais scraper ne sont pas interdites. Puis, j'ai choisi d'utiliser une bibiothèque qui permet de faire le web scraping et web crawbling à la fois. Il s'appelle craw4ai, une bibliothèque de Web Crawler & Scraper qui ont des API pour LLM, pour plus d'information, veuillez consulter le lien officiel : https://docs.crawl4ai.com/. Par rapport aux outiles traditionels, il permet d'offrir une variété de formats de output par exemple le markdown. Et il permet aussi d'utiliser l'IA pour extraire des données dans les résultats, ce qui généralise bcp le script. On a pas besoin de comprendre la structure HTML du site qu'on va faire le scraping et crawling, au lieu de la parser par nous même, ce qui laisse notre script moins généraliste, cet outil permet de donner les instructions à l'IA et le demander d'extraire les informations pour nous. Dans notre cas, c'est pratique, car je exm'intéresse seulement aux certains contenus de la page.
## TP3
J'ai fait les test de zipf, l'analyse de phrase, le calcul des mots rares et le test de la diversité de vocabulaire. Étant donné que notre corpus est un simple example, j'ai pas collecté assez de données. Mais je les augementerai dans l'étape 4.
## TP4
Dans cette étape, j'ai développé le script `enhance_corpus` pour augementer le corpus par des méthodes suivantes : le remplacement de synonyme, création de nouvelle phrases en changeant l'ordre des mots et avec les modèles prétrainés. J'ai choisi d'utiliser le GPT-2 pour comme le modèle de NLG à entraîner vu que j'ai pas un très grand corpus.
## TP5
Pour l'étape fine-tuning, j'ai utilisé le modèle `asi/gpt-fr-cased-small` qui contient à la taille de mon corpus.
## TP6
Pour cette étape, j'ai testé la perplexité du modèle
  
