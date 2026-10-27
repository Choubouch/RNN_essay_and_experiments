### Mathis Verdan (MVA)

# Essai sur l'article de blog : The Unreasonable Effectiveness of Recurrent Neural Networks 

## I - Introduction

Depuis 2017, avec la publication de l'article [_Attention is all you need_](https://arxiv.org/abs/1706.03762), l'architecture de _transformer_ est devenue standard dans le monde des _LLMs_. Le mécanisme d'attention est aujourd'hui largement répandu, mais il n'en a pas toujours été ainsi. 

Cet article de blog intitulé [_The Unreasonable Effectiveness of Recurrent Neural Networks_](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) et publié en mai 2015 par Andrej Karpathy (ancien chercheur chez _OpenAI_) montre qu'il y a seulement 11 ans, les _transformers_ n'avaient pas encore émergés, et des architectures alternatives comme les _Recurrent Neural Networks_ (RNNs) étaient envisagées. 

Dans son blog, A. Karpathy présente le fonctionnement général des RNNs, ainsi que quelques tests qu'il a réalisés pour démontrer la grande efficacité de cette architecture dans diverses situations. 

Cet article a retenu mon attention pour plusieurs raisons : il m'a d'abord semblé intéressant de voir l'état de la recherche sur les LLMs au début de leur démocratisation, ensuite A. Karpathy explique dès le début que les RNNs permettent d'avoir des résultats bluffants avec de petits modèles pouvant tourner en local, et enfin je n'ai jamais eu l'occasion de m'intéresser en détail aux RNNs, et cela me semblait important pour ma culture personnelle. 


## II - Un article intéressant

### Un modèle simple

J'ai tout d'abord trouvé que l'architecture de ce modèle était plutôt simple en comparaison avec les résultats complexes auxquels elle donne lieu. L'idée de base du RNN consiste à appliquer récursivement une fonction $f$ sur une séquence de données $(x_t)$. Cette fonction produit une représentation intermédiaire ou cachée $h_t$ utilisée par le réseau pour faire ses prédictions. En notant $\theta$ les paramètres du réseau, on a $h_{(t+1)} = f(h_t, x_t, \theta) = \tanh(W h_t + U x_t + b)$. Ce qui est intéressant ici est que la fonction $f$ est la même à chaque étape de la récursion. Cela permet au réseau de traiter des données de taille variable puisqu'il suffit juste d'appliquer la fonction $f$ autant de fois que nécessaire. 

Cette apparente simplicité des RNNs peut malgré tout donner des résultats complexes. En effet, il peut sembler contre-intuitif de pouvoir générer un texte cohérent avec une fonction qui ne prend en paramètre que le token courant $x_t$. Mais en réalité, l'état caché $h_t$ intègre l'ensemble du texte déjà généré, et cela permet au RNN d'avoir une mémoire. Ainsi, même si le RNN se contente de prédire la prochaine lettre, le texte a une cohérence globale.

Pour augmenter la complexité et la puissance du modèle, il est possible de combiner deux RNNs pour faire un réseau à deux couches (cela consiste à connecter la sortie du premier réseau à l'entrée du second), et l'auteur conseille également d'utiliser une architecture _Long Short-Term Memory (LSTM)_ pour limiter le phénomène de gradient évanescent lors de l'entraînement (la matrice $W$ multiplie le vecteur $h$ plusieurs fois d'affilée, ce qui fait que l'information passée tend exponentiellement vers $0$ et est donc oubliée). 

### Apprentissage de relations complexes

Malgré une architecture assez simple, et une réputation de modèle difficile à entrainer, l'auteur soutient l'inverse dans son blog. Le LSTM entraîné est capable d'appendre une langue et des règles de syntaxe assez rapidement (il peut s'agir d'une langue "humaine" comme l'anglais, ou bien un langage informatique). Le fait que le programme apprenne la structure de la langue au fur et à mesure sans avoir reçu d'indication m'a semblé particulièrement intéressant. Le modèle commence par apprendre les règles les plus simples (taille des mots, ponctuation ...), avant d'approfondir sa "compréhension" de la langue (accords, conjugaison ...). Pour cela, même si aucune indication n'est donnée, certains neurones se spécialisent et deviennent humainement interprétables (détection des parenthèses, des urls ...). On remarque ainsi qu'une structure logique émerge des données, sans connaissance préalable. Pour satisfaire l'objectif de prédiction du prochain caractère, il est ainsi intéressant pour le RNN de retenir et de structurer certaines informations dans son état caché.

## III - Des détails surprenants
### Absence de _tokenizer_ et petits datasets

Selon moi, un des points les plus surprenants des résultats montrés est la capacité du RNN à avoir de bonnes performances malgré une absence de connaissances _a priori_ du langage appris. L'auteur n'utilise pas de _tokenizer_, ni d'_embbedings_ pré-entrainés, mais donne directement l'encodage _one-hot_ des lettres au RNN. Cela complexifie la tâche à réaliser puisque certaines connaissances que l'on a du langage (par exemple : on trouve rarement un "w" après un "x" dans les mots anglais) ne sont pas données au RNN pendant l'entraînement. Le modèle doit donc apprendre simultanément des structures à différentes échelles : à petite échelle (enchaînement valide de lettres), à moyenne échelle (taille des mots, enchaînement sujet-verbe ...) et à plus grande échelle (création de paragraphes, fermeture de parenthèse ...). 

Enfin, la taille des datasets utilisés m'a également parue relativement faible. Dans le cas des essais de Paul Graham ou des textes de Shakespeare, les datasets ne faisaient que quelques MB, ce qui permet malgré tout au modèle d'apprendre la langue et le style d'écriture.

### Les RNNs sont de bons menteurs

Les LLMs sont connus pour générer des textes convaincants en apparence, mais faux dans le fond. J'ai trouvé cela amusant de constater que le même phénomène se produit avec les RNNs présentés dans le blog. En dépit d'une apparente compréhension de la langue par le modèle, on s'aperçoit vite que le texte généré n'a pas vraiment de sens. J'ai ainsi été étonné par la capacité des modèles à recréer une distribution de caractères cohérente, sans pour autant que cela témoigne d'une compréhension sémantique du texte.

## IV - Idées d'expérimentations

### Représentation du texte et performances

Une expérience qui pourrait prolonger ce travail serait d'étudier l'effet de la représentation du texte sur les performances du RNN. L'utilisation de tokens pourrait simplifier l'apprentissage en injectant plus de connaissances sur le langage appris. Pour vérifier cela, il faudrait entraîner le même modèle sur les mêmes données, mais en ajoutant une étape de prétraitement pour regrouper certains caractères entre eux (_byte pair encoding_).  

Pour mesurer l'impact de l'utilisation de tokens sur les performances, il serait intéressant de suivre l'évolution de mesures statistiques (log-vraisemblance ou perplexité), mais aussi d'essayer de mesurer si le texte généré est sémantiquement correct (et n'a pas juste une apparence de texte correct). Cela devrait par exemple être possible en demandant à un LLM plus performant si les phrases générées ont du sens. La durée d'entraînement pourrait également montrer un éventuel gain d'efficacité. 


## V - Conclusion

Cet article démontre qu'une tâche simple et locale, permet au RNN de générer des exemples convaincants avec une structure globale complexe. L'architecture récurrente du modèle lui permet de mémoriser certaines informations au fur et à mesure, ce qui explique probablement une partie de son succès. Toutefois, générer des exemples convaincants ne montre pas non plus que le modèle "comprend" réellement ce qu'il produit. Enfin, malgré un modèle performant, il est intéressant de noter que cela n'a pas empêché aux _transformers_ de s'imposer par la suite.

PS : pour test l'efficacité des RNNs par moi-même, suivre les préceptes de Confucius, et par esprit patriotique, j'ai implémenté dans [ce repo](https://github.com/Choubouch/RNN_essay_and_experiments) un modèle permettant de générer des noms de communes françaises. 