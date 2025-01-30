import os
import pytest

def test_historial_csv():
    file_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'historial.csv')
    assert os.path.isfile(file_path), f"Archivo de resultados no existe en {file_path}."

if __name__ == "__main__":
    test_historial_csv()