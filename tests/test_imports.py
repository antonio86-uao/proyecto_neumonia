def test_imports():
    try:
        from src.data import read_img, preprocess_img
        from src.models import load_model, grad_cam
        from src.interface import integrator
        print("Importaciones exitosas")
        return True
    except Exception as e:
        print(f"Error en importaciones: {str(e)}")
        return False

if __name__ == "__main__":
    test_imports()