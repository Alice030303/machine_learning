"""
Script pour visualiser l'augmentation des données
"""

from preprocessing import visualize_augmentation

# Utiliser le répertoire de données
DATA_DIR = "data"

# Visualiser l'augmentation
visualize_augmentation(
    data_dir=DATA_DIR,
    target_size=(224, 224),
    num_samples=4  # Nombre d'exemples à afficher
)
