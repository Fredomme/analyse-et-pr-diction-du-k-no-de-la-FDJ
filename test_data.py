# tests/test_data.py

import pytest
from data_loader import load_keno_data

def test_load_keno_data():
    """
    Vérifie que load_keno_data() renvoie un DataFrame non vide,
    avec les colonnes attendues, etc.
    """
    df = load_keno_data()
    assert not df.empty, "Le DataFrame est vide!"
    assert "date_de_tirage" in df.columns, "date_de_tirage manquante"
    # etc. On peut faire d'autres checks
