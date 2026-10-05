#!/bin/bash
# Se place à la racine du projet (WalkIA)
cd "$(dirname "$0")/.."

# Exécute l'entraînement rapide
env/bin/python src/train.py
