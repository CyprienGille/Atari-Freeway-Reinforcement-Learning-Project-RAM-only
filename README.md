# Projet de Reinforcement Learning - Freeway RAM - Architecture de l'ANN

Ce repository contient un projet de RL consistant à essayer d'entraîner un agent (Artificial Neural Network, ANN (terme à utiliser imposé par les consignes)) à jouer au jeu Freeway d'Atari, implémenté dans gym. Selon les consignes du projet, l'agent doit apprendre uniquement à partir de la RAM du jeu, ce qui ne donnera pas de très bonnes performances. Pour (beaucoup) plus de détails, voir `Rapport Freeway RL.pdf`.

---

## Fichiers

- up_runner.py : Faire jouer un agent qui ne fait qu'aller vers le haut
- data_creation.py : Créer la database en jouant
- ann_training.py : Entrainer un réseau de neurones sur la database, et le sauvegarder
- ann_playing.py : Faire jouer un agent sauvegardé à Freeway
- architecture_tests.py : Faire des tests sur l'influence de la topologie de l'ANN sur le score, la vitesse de convergence, et la précision
