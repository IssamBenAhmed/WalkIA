#!/bin/bash
# Se place à la racine du projet (WalkIA)
cd "$(dirname "$0")/.."

# Exécute le rendu visuel
DISPLAY=:0 MON_ROBOT_RENDER=1 env/bin/python src/train.py
