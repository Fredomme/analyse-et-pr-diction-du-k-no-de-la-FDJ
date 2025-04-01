.PHONY: setup train eval run clean

setup:
	@echo "🔧 Création de l'environnement virtuel et installation des dépendances..."
	@if [ ! -d "keno_env" ]; then python3 -m venv keno_env; fi
	@. keno_env/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

train:
	@echo "🚀 Entraînement des modèles..."
	@. keno_env/bin/activate && python3 main_train.py

eval:
	@echo "📊 Évaluation du modèle..."
	@. keno_env/bin/activate && python3 main_eval.py

run:
	@echo "🏁 Lancement du pipeline complet..."
	@. keno_env/bin/activate && python3 main.py

clean:
	@echo "🧹 Nettoyage des fichiers inutiles..."
	@rm -rf __pycache__ *.pyc models_out/ eval_log.csv train_log_*.csv preproc_log.csv
