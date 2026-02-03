"""
Prétraitement des images pour la reconnaissance d'animaux
- Redimensionnement uniforme des images
- Normalisation des valeurs de pixels
- Augmentation des données : rotation, zoom, retournement
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Dossiers à ignorer lors du traitement
IGNORED_FOLDERS = ['extra']


class ImagePreprocessor:
    """Classe pour le prétraitement des images"""
    
    def __init__(self, target_size=(224, 224), normalize=True):
        """
        Initialise le préprocesseur d'images
        
        Args:
            target_size: Tuple (height, width) pour le redimensionnement uniforme
            normalize: Booléen pour normaliser les valeurs de pixels (0-1)
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def resize_image(self, image_path):
        """
        Redimensionne une image de manière uniforme
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image redimensionnée (numpy array)
        """
        img = Image.open(image_path)
        img = img.resize(self.target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img)
        return img_array
    
    def normalize_pixels(self, image):
        """
        Normalise les valeurs de pixels entre 0 et 1
        
        Args:
            image: Image sous forme de numpy array
            
        Returns:
            Image normalisée
        """
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        # Normalisation entre 0 et 1
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    
    def preprocess_single_image(self, image_path):
        """
        Prétraite une seule image (redimensionnement + normalisation)
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image prétraitée
        """
        img = self.resize_image(image_path)
        if self.normalize:
            img = self.normalize_pixels(img)
        return img


def get_valid_classes(data_dir):
    """
    Récupère la liste des classes valides (dossiers) en excluant les dossiers ignorés
    
    Args:
        data_dir: Répertoire contenant les sous-dossiers par classe
        
    Returns:
        Liste des noms de classes valides
    """
    valid_classes = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        # Ignorer les dossiers dans IGNORED_FOLDERS et ne garder que les dossiers
        if os.path.isdir(item_path) and item.lower() not in [f.lower() for f in IGNORED_FOLDERS]:
            valid_classes.append(item)
    return sorted(valid_classes)


def create_data_generators(data_dir, target_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Crée les générateurs de données avec augmentation pour l'entraînement
    et sans augmentation pour la validation
    
    Args:
        data_dir: Répertoire contenant les sous-dossiers par classe
        target_size: Taille cible pour le redimensionnement (height, width)
        batch_size: Taille des lots
        validation_split: Proportion des données pour la validation
        
    Returns:
        train_generator, validation_generator, class_names
    """
    
    # Récupérer les classes valides (exclure 'extra' et autres dossiers ignorés)
    valid_classes = get_valid_classes(data_dir)
    
    if not valid_classes:
        raise ValueError(f"Aucune classe valide trouvée dans {data_dir}. "
                        f"Assurez-vous que les dossiers de classes existent et ne sont pas dans {IGNORED_FOLDERS}")
    
    # Augmentation des données pour l'entraînement
    train_datagen = ImageDataGenerator(
        # Normalisation des valeurs de pixels (0-1)
        rescale=1./255,
        
        # Augmentation : rotation
        rotation_range=359,  # Rotation aléatoire jusqu'à 359 degrés
        
        # Augmentation : zoom
        zoom_range=0.2,  # Zoom aléatoire jusqu'à 20%
        
        # Augmentation : retournement horizontal
        horizontal_flip=True,
        
        # Retournement vertical (optionnel)
        vertical_flip=False,
        
        # Translation (déplacement)
        width_shift_range=0.1,  # Déplacement horizontal jusqu'à 10%
        height_shift_range=0.1,  # Déplacement vertical jusqu'à 10%
        
        # Remplissage pour les transformations
        fill_mode='nearest',
        
        # Validation split
        validation_split=validation_split
    )
    
    # Pas d'augmentation pour la validation (seulement normalisation)
    validation_datagen = ImageDataGenerator(
        rescale=1./255,  # Seulement la normalisation
        validation_split=validation_split
    )
    
    # Générateur d'entraînement avec augmentation
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,  # Redimensionnement uniforme
        batch_size=batch_size,
        class_mode='categorical',
        classes=valid_classes,  # Spécifier explicitement les classes à inclure
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Générateur de validation sans augmentation
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=target_size,  # Redimensionnement uniforme
        batch_size=batch_size,
        class_mode='categorical',
        classes=valid_classes,  # Spécifier explicitement les classes à inclure
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Récupération des noms de classes
    class_names = list(train_generator.class_indices.keys())
    
    print(f"\n✓ Générateurs de données créés avec succès!")
    print(f"  - Classes détectées: {class_names}")
    if IGNORED_FOLDERS:
        print(f"  - Dossiers ignorés: {IGNORED_FOLDERS}")
    print(f"  - Nombre d'images d'entraînement: {train_generator.samples}")
    print(f"  - Nombre d'images de validation: {validation_generator.samples}")
    print(f"  - Taille des images: {target_size}")
    print(f"  - Taille des lots: {batch_size}")
    
    return train_generator, validation_generator, class_names


def visualize_augmentation(data_dir, target_size=(224, 224), num_samples=4):
    """
    Visualise les effets de l'augmentation des données
    
    Args:
        data_dir: Répertoire contenant les images
        target_size: Taille cible pour le redimensionnement
        num_samples: Nombre d'exemples à visualiser
    """
    # Récupérer les classes valides (exclure 'extra' et autres dossiers ignorés)
    valid_classes = get_valid_classes(data_dir)
    
    # Créer un générateur avec augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=359,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )
    
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=1,
        class_mode='categorical',
        classes=valid_classes,  # Spécifier explicitement les classes à inclure
        shuffle=True
    )
    
    # Afficher quelques exemples
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle('Exemples d\'augmentation de données', fontsize=16)
    
    for i in range(num_samples):
        # Image originale (première ligne)
        img, label = next(generator)  # Utiliser next() au lieu de .next()
        axes[0, i].imshow(img[0])
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        # Image augmentée (deuxième ligne)
        img_aug, _ = next(generator)  # Utiliser next() au lieu de .next()
        axes[1, i].imshow(img_aug[0])
        axes[1, i].axis('off')
        axes[1, i].set_title('Augmentée')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualisation de l'augmentation sauvegardée dans 'augmentation_examples.png'")
    plt.show()


def preprocess_dataset(data_dir, output_dir=None, target_size=(224, 224)):
    """
    Prétraite toutes les images du dataset et les sauvegarde (optionnel)
    
    Args:
        data_dir: Répertoire contenant les sous-dossiers par classe
        output_dir: Répertoire de sortie (si None, pas de sauvegarde)
        target_size: Taille cible pour le redimensionnement
    """
    preprocessor = ImagePreprocessor(target_size=target_size, normalize=True)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    total_images = 0
    
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        
        # Ignorer les dossiers dans IGNORED_FOLDERS
        if class_folder.lower() in [f.lower() for f in IGNORED_FOLDERS]:
            print(f"\n⚠ Dossier ignoré: {class_folder}")
            continue
        
        if not os.path.isdir(class_path):
            continue
        
        if output_dir:
            output_class_dir = os.path.join(output_dir, class_folder)
            os.makedirs(output_class_dir, exist_ok=True)
        
        print(f"\nTraitement de la classe: {class_folder}")
        class_images = 0
        
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # Prétraitement
                    processed_img = preprocessor.preprocess_single_image(img_path)
                    
                    if output_dir:
                        # Sauvegarder l'image prétraitée
                        output_path = os.path.join(output_class_dir, img_file)
                        img_save = (processed_img * 255).astype(np.uint8)
                        Image.fromarray(img_save).save(output_path)
                    
                    class_images += 1
                    total_images += 1
                    
                except Exception as e:
                    print(f"  Erreur avec {img_file}: {e}")
        
        print(f"  ✓ {class_images} images traitées pour la classe '{class_folder}'")
    
    print(f"\n✓ Prétraitement terminé: {total_images} images au total")


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data"
    TARGET_SIZE = (224, 224)  # Taille standard pour les modèles pré-entraînés
    BATCH_SIZE = 32
    
    print("=" * 60)
    print("PRÉTRAITEMENT DES IMAGES")
    print("=" * 60)
    print(f"\nRépertoire des données: {DATA_DIR}")
    print(f"Taille cible: {TARGET_SIZE}")
    print(f"Taille des lots: {BATCH_SIZE}")
    
    # Vérifier que le répertoire existe
    if not os.path.exists(DATA_DIR):
        print(f"\n⚠ Erreur: Le répertoire '{DATA_DIR}' n'existe pas!")
        print("   Créez le répertoire et ajoutez vos images dans des sous-dossiers:")
        print("   - data/tigres/")
        print("   - data/elephants/")
        print("   - data/giraffes/")
    else:
        # Créer les générateurs de données avec augmentation
        # Les images sont prétraitées à la volée (en mémoire) pendant l'entraînement
        train_gen, val_gen, classes = create_data_generators(
            data_dir=DATA_DIR,
            target_size=TARGET_SIZE,
            batch_size=BATCH_SIZE,
            validation_split=0.2
        )
        
        print("\n" + "=" * 60)
        print("✓ Prétraitement configuré avec succès!")
        print("=" * 60)
        print("\nLes générateurs sont prêts pour l'entraînement du modèle.")
        print("Les images sont prétraitées à la volée (redimensionnement + normalisation)")
        print("et l'augmentation est appliquée dynamiquement pendant l'entraînement.")
        print("\nPour visualiser l'augmentation, exécutez:")
        print("  visualize_augmentation(DATA_DIR)")
