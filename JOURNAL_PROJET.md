# Journal du Projet WalkIA - Robot Quadrupède "Canard"

## Ce qui a été accompli (Dernière session)

1. **Système de Logs Avancé** :
   - Mise en place d'un tableau de bord professionnel `stats_robot.csv` (géré par Pandas).
   - Affichage toutes les 10 secondes dans le terminal avec des métriques claires : survie, hauteur, réussite >2s/5s, distance, vitesse, taux de chute.

2. **Résolution du "Bug de Naissance"** :
   - Découverte majeure via l'analyse du CSV : le robot mourait à la frame 1 (0.04s) car on utilisait `qpos[1]` (l'offset au lieu de la hauteur absolue).
   - Intégration de la méthode robuste utilisant l'API native MuJoCo (`self.model.body("torso").id` et `self.data.xpos`) pour lire l'altitude exacte du torse dans le repère du monde.

3. **Correction du Minimum Local (Anti-Cheat "Grand Écart")** :
   - Le robot avait appris à se jeter sur les genoux (hauteur de 0.4m) pour éviter la chute sans effort.
   - Augmentation du seuil critique de "Game Over" de 0.3m à 0.55m pour forcer l'usage des mollets et des palmes.

4. **Intégration de la Récompense "Stabilité d'abord" (Merci au camarade !)** :
   - Refonte totale du système de récompense dans `mon_env.py` :
     - **Posture saine** : Zone de confort récompensée entre 0.7m et 0.9m de hauteur (+5 points).
     - **Appui Palmes** : Détection physique via les capteurs de contact pour récompenser l'utilisation des palmes.
     - **Punition Genoux** : Pénalité stricte (-12 points) si le robot triche en utilisant ses genoux.
     - **Rendu Visuel** : Désactivation du Game Over immédiat en mode `human` pour permettre le débogage visuel.

## Ce qu'il reste à faire (Prochaine session : Demain)

1. **Curriculum Learning (L'étape reine)** :
   - Mettre en place un système d'apprentissage progressif en 3 phases distinctes (plutôt que d'attendre tout en même temps) :
     1. **Phase 1 : Stabilité** (Garder le robot debout sur place avec une excellente posture).
     2. **Phase 2 : Marche lente** (Introduire la vitesse sans perdre l'équilibre).
     3. **Phase 3 : Trot** (Accélérer et optimiser la foulée).

2. **Implémentation technique du Curriculum** :
   - Décider de la manière d'activer les phases (soit en modifiant les poids des récompenses dans `mon_env.py` dynamiquement selon le nombre d'épisodes, soit avec un script maître).
   - Intégrer les formules/codes fournis par le camarade pour ces 3 phases.
