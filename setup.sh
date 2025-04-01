#!/bin/bash

echo "🔧 Initialisation de l'environnement virtuel Python..."

# Crée le venv s'il n'existe pas
if [ ! -d "keno_env" ]; then
    python3 -m venv keno_env
    echo "✅ Environnement virtuel créé : keno_env"
else
    echo "ℹ️ Environnement déjà existant"
fi

# Active le venv
source keno_env/bin/activate
echo "🐍 Environnement activé"

# Installe les dépendances
echo "📦 Installation des packages depuis requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Lancement ou instructions
echo -e "\n✅ Setup terminé !"
echo "➡️ Tu peux maintenant lancer : python3 main.py"
