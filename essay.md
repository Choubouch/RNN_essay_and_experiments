### Mathis Verdan (MVA), 19 octobre 2026

# Essai sur l'article de blog : The Unreasonable Effectiveness of Recurrent Neural Networks 

## Introduction

Depuis 2017, avec la publication de l'article [_Attention is all you need_](https://arxiv.org/abs/1706.03762), puis avec le développement des modèles GPT, l'architecture _Transformer_ est devenue standard dans le monde des LLM. Le mécanisme d'attention apparaît également comme évident aujourd'hui, à tel point qu'il est maintenant enseigné dans les universités, mais il n'en a pas toujours été ainsi. 

Cet article de blog intitulé [_The Unreasonable Effectiveness of Recurrent Neural Networks_](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) et publié en mai 2015 par Andrej Karpathy (ancien chercheur chez _OpenAI_) montre qu'il y a seulement 11 ans, les techniques "modernes" de conception et d'entraînement de LLMs étaient encore au début de leur développement. Ainsi, à une période où les Transformers n'avaient pas encore émergés, des architectures alternatives comme les _Recurrent Neural Networks_ (RNNs) étaient envisagées. 

Dans son blog, A. Karpathy présente le fonctionnement général des RNNs, ainsi que quelques tests qu'il a réalisés pour démontrer la grande efficacité de cette architecture dans diverses situations. 

Cet article a retenu mon attention pour plusieurs raisons : il m'a d'abord semblé intéressant de voir l'état de la recherche sur les LLMs au début de leur démocratisation (et notamment avant les modèles GPT), ensuite A. Karpathy explique dès le début que les RNNs permettent d'avoir des résultats bluffants avec de petits modèles pouvant tourner en local, et enfin je n'ai jamais eu l'occasion de m'intéresser en détail aux RNNs, et cela me semblait important pour ma culture personnelle. 


## Pourquoi cet article a retenu mon attention

- des règles simples donnent un comportement complexe
- connexion entre les neurones pour donner un comportement cx
- des techniques de visualisation
- le fonctionnement des Rnn
- pas si difficile a entrainer

### Un modèle simple
N'ayant jamais eu l'occasion de m'intéresser en détail aux RNNs, j'ai tout d'abord trouvé que l'architecture de ce modèle était plutôt simple, en comparaison avec les résultats assez complexes auxquels elle donne lieu. L'idée de base du RNN consiste à appliquer récursivement une fonction $f$ sur une séquence de données $(x_t)$. Cette fonction produit une représentation intermédiaire ou cachée $h_t$ utilisée par le réseau pour faire ces prédictions. En notant $\theta$ les paramètres du réseau, on a $h_{(t+1)} = f(h_t, x_t, \theta) = \tanh(W h_t + U x_t + b)$ en général (les paramètres à apprendre sont les matrices $W$, $U$ et le biais $b$). Ce qui est intéressant ici est que la fonction $f$ est la même à chaque étape de la récursion. Cela permet au réseau de traiter des données de taille variable puisqu'il suffit juste d'appliquer la fonction $f$ autant de fois que nécessaire. 

Cette apparente simplicité des RNNs peut malgré tout donner des résultats complexes. En effet, il peut sembler contre-intuitif de pouvoir générer un texte cohérent avec une fonction qui ne prend en paramètre que le token courant $x_t$. Mais en réalité, l'état caché $h_t$ intègre l'ensemble du texte déjà généré, et cela permet au RNN d'avoir une mémoire. C'est pourquoi, dans l'exemple donné par A. Karpathy, un RNN qui doit générer le mot "hello" lettre par lettre est capable de prédire la lettre "l" après le premier "l", puis la lettre "o" après le second. 

Pour augmenter la complexité et la puissance du modèle, il est possible de combiner deux RNNs pour faire un réseau à deux couches (cela consiste à connecter la sortie du premier réseau à l'entrée du second), et l'auteur conseille également d'utiliser une architecture _Long Short-Term Memory (LSTM)_ pour limiter le phénomène de gradient évanescent lors de l'entraînement (la matrice $W$ multiplie le vecteur $h$ plusieurs fois d'affilée, ce qui fait que l'information passée tend exponentiellement vers $0$ ou explose au bout de quelques itérations). 

## La surprenante simplicité des RNNs
- pas de tokeniser
- petits datasets 
- pas d'attention
- il bullshit bien

## Idées d'expérimentations