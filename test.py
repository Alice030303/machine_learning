"""
Script de test pour évaluer le modèle CNN sur les images de test
- Charge le modèle depuis models/best_model.h5
- Teste les images dans data/test/
- Les images commencent par E (elephant), G (giraffe), T (tigre)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from preprocessing import ImagePreprocessor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def load_model(model_path="models/best_model.h5"):
    """
    Charge le modèle sauvegardé
    
    Args:
        model_path: Chemin vers le fichier du modèle
        
    Returns:
        Modèle chargé
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle n'existe pas : {model_path}")
    
    print(f"Chargement du modele depuis {model_path}...")
    model = keras.models.load_model(model_path)
    print("OK - Modele charge avec succes!")
    
    return model


def get_class_from_filename(filename):
    """
    Détermine la classe réelle à partir du nom de fichier
    E -> elephant, G -> giraffe, T -> tigre
    
    Args:
        filename: Nom du fichier
        
    Returns:
        Nom de la classe
    """
    first_char = filename[0].upper()
    
    if first_char == 'E':
        return 'elephant'
    elif first_char == 'G':
        return 'giraffe'
    elif first_char == 'T':
        return 'tigre'
    else:
        raise ValueError(f"Impossible de déterminer la classe pour {filename}. "
                       f"Le nom doit commencer par E, G ou T.")


def load_test_images(test_dir="data/test", target_size=(224, 224)):
    """
    Charge et prétraite toutes les images de test
    
    Args:
        test_dir: Répertoire contenant les images de test
        target_size: Taille cible pour le redimensionnement
        
    Returns:
        images: Liste des images prétraitées (numpy arrays)
        filenames: Liste des noms de fichiers
        true_labels: Liste des labels réels
    """
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Le répertoire de test n'existe pas : {test_dir}")
    
    preprocessor = ImagePreprocessor(target_size=target_size, normalize=True)
    
    images = []
    filenames = []
    true_labels = []
    
    print(f"\nChargement des images depuis {test_dir}...")
    
    # Obtenir tous les fichiers d'image
    image_files = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        raise ValueError(f"Aucune image trouvée dans {test_dir}")
    
    # Trier les fichiers pour un ordre cohérent
    image_files.sort()
    
    for filename in image_files:
        image_path = os.path.join(test_dir, filename)
        
        try:
            # Prétraiter l'image
            processed_img = preprocessor.preprocess_single_image(image_path)
            
            # Obtenir la classe réelle
            true_class = get_class_from_filename(filename)
            
            images.append(processed_img)
            filenames.append(filename)
            true_labels.append(true_class)
            
        except Exception as e:
            print(f"ATTENTION - Erreur lors du traitement de {filename}: {e}")
            continue
    
    print(f"OK - {len(images)} images chargees avec succes")
    
    return np.array(images), filenames, true_labels


def predict_images(model, images):
    """
    Fait des prédictions sur les images
    
    Args:
        model: Modèle CNN
        images: Tableau numpy d'images prétraitées
        
    Returns:
        predictions: Prédictions du modèle (probabilités)
        predicted_classes: Classes prédites (indices)
    """
    print("\nGeneration des predictions...")
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    print("OK - Predictions terminees!")
    
    return predictions, predicted_classes


def get_class_names():
    """
    Retourne les noms de classes dans l'ordre attendu par le modèle
    (ordre alphabétique des dossiers: elephant, giraffe, tigre)
    
    Returns:
        Liste des noms de classes
    """
    return ['elephant', 'giraffe', 'tigre']


def evaluate_predictions(true_labels, predicted_classes, class_names):
    """
    Évalue les prédictions et affiche les résultats
    
    Args:
        true_labels: Labels réels
        predicted_classes: Classes prédites (indices)
        class_names: Liste des noms de classes
    """
    # Convertir les labels réels en indices
    label_to_index = {name: idx for idx, name in enumerate(class_names)}
    true_indices = [label_to_index[label] for label in true_labels]
    
    # Calculer l'accuracy
    accuracy = accuracy_score(true_indices, predicted_classes)
    
    print("\n" + "=" * 60)
    print("RESULTATS DE L'EVALUATION")
    print("=" * 60)
    print(f"\nPrecision globale : {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Rapport de classification
    print("\nRapport de classification :")
    print(classification_report(
        true_indices, 
        predicted_classes, 
        target_names=class_names,
        digits=4
    ))
    
    # Matrice de confusion
    cm = confusion_matrix(true_indices, predicted_classes)
    
    print("\nMatrice de confusion :")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe predite')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("OK - Matrice de confusion sauvegardee dans 'confusion_matrix.png'")
    plt.show()
    
    return accuracy, cm


def display_detailed_results(filenames, true_labels, predicted_classes, predictions, class_names):
    """
    Affiche les résultats détaillés pour chaque image
    
    Args:
        filenames: Liste des noms de fichiers
        true_labels: Labels réels
        predicted_classes: Classes prédites (indices)
        predictions: Probabilités de prédiction
        class_names: Liste des noms de classes
    """
    print("\n" + "=" * 60)
    print("RESULTATS DETAILLES PAR IMAGE")
    print("=" * 60)
    
    correct = 0
    incorrect = 0
    
    for i, filename in enumerate(filenames):
        true_label = true_labels[i]
        pred_idx = predicted_classes[i]
        pred_label = class_names[pred_idx]
        confidence = predictions[i][pred_idx] * 100
        
        is_correct = (true_label == pred_label)
        status = "[OK]" if is_correct else "[X]"
        
        if is_correct:
            correct += 1
        else:
            incorrect += 1
        
        print(f"\n{status} {filename}")
        print(f"   Vraie classe : {true_label}")
        print(f"   Prediction   : {pred_label} ({confidence:.2f}%)")
        
        # Afficher toutes les probabilités
        print(f"   Probabilites :")
        for j, class_name in enumerate(class_names):
            prob = predictions[i][j] * 100
            marker = " <--" if j == pred_idx else ""
            print(f"     - {class_name}: {prob:.2f}%{marker}")
    
    print("\n" + "=" * 60)
    print(f"Résumé : {correct} correctes, {incorrect} incorrectes")
    print("=" * 60)


def visualize_predictions(filenames, true_labels, predicted_classes, predictions, 
                         class_names, test_dir="data/test", num_samples=9):
    """
    Visualise quelques prédictions avec les images
    
    Args:
        filenames: Liste des noms de fichiers
        true_labels: Labels réels
        predicted_classes: Classes prédites (indices)
        predictions: Probabilités de prédiction
        class_names: Liste des noms de classes
        test_dir: Répertoire des images de test
        num_samples: Nombre d'images à visualiser
    """
    print(f"\nVisualisation de {min(num_samples, len(filenames))} predictions...")
    
    num_samples = min(num_samples, len(filenames))
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        filename = filenames[i]
        image_path = os.path.join(test_dir, filename)
        
        # Charger et afficher l'image
        img = plt.imread(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Informations de prédiction
        true_label = true_labels[i]
        pred_idx = predicted_classes[i]
        pred_label = class_names[pred_idx]
        confidence = predictions[i][pred_idx] * 100
        
        is_correct = (true_label == pred_label)
        color = 'green' if is_correct else 'red'
        status = "[OK]" if is_correct else "[X]"
        
        title = f"{status} {filename}\n"
        title += f"Vrai: {true_label}\n"
        title += f"Predit: {pred_label} ({confidence:.1f}%)"
        
        axes[i].set_title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=150, bbox_inches='tight')
    print("OK - Visualisations sauvegardees dans 'test_predictions.png'")
    plt.show()


def main():
    """
    Fonction principale pour tester le modèle
    """
    print("=" * 60)
    print("TEST DU MODELE CNN")
    print("=" * 60)
    
    # Configuration
    MODEL_PATH = "models/best_model.h5"
    TEST_DIR = "data/test"
    TARGET_SIZE = (224, 224)
    
    try:
        # 1. Charger le modèle
        model = load_model(MODEL_PATH)
        
        # 2. Obtenir les noms de classes
        class_names = get_class_names()
        print(f"\nClasses du modele : {class_names}")
        
        # 3. Charger les images de test
        images, filenames, true_labels = load_test_images(TEST_DIR, TARGET_SIZE)
        
        # 4. Faire les prédictions
        predictions, predicted_classes = predict_images(model, images)
        
        # 5. Convertir les indices en noms de classes
        predicted_labels = [class_names[idx] for idx in predicted_classes]
        
        # 6. Évaluer les résultats
        accuracy, cm = evaluate_predictions(true_labels, predicted_classes, class_names)
        
        # 7. Afficher les résultats détaillés
        display_detailed_results(filenames, true_labels, predicted_classes, 
                               predictions, class_names)
        
        # 8. Visualiser quelques prédictions
        visualize_predictions(filenames, true_labels, predicted_classes, 
                            predictions, class_names, TEST_DIR)
        
        print("\n" + "=" * 60)
        print("OK - Test termine avec succes!")
        print("=" * 60)
        print("\nFichiers generes :")
        print("  - confusion_matrix.png : Matrice de confusion")
        print("  - test_predictions.png : Visualisations des predictions")
        
    except Exception as e:
        print(f"\nERREUR : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
