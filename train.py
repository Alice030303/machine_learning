"""
Étape 4 : Entraînement et validation du modèle CNN
- Séparation des données (train/validation)
- Optimisation avec Adam et entropie croisée
- Prévention du surapprentissage (dropout, augmentation, callbacks)
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from preprocessing import create_data_generators
from model import build_cnn_model, compile_model, create_callbacks


def train_model(
    data_dir="data",
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    learning_rate=0.001,
    save_best_model=True,
    model_save_path="models/best_model.h5"
):
    """
    Entraîne le modèle CNN pour la reconnaissance d'animaux
    
    Args:
        data_dir: Répertoire contenant les images par classe
        epochs: Nombre d'époques d'entraînement
        batch_size: Taille des lots
        validation_split: Proportion des données pour la validation (20-30%)
        learning_rate: Taux d'apprentissage
        save_best_model: Sauvegarder le meilleur modèle
        model_save_path: Chemin pour sauvegarder le modèle
        
    Returns:
        model: Modèle entraîné
        history: Historique d'entraînement
    """
    
    print("=" * 60)
    print("ENTRAÎNEMENT DU MODÈLE CNN")
    print("=" * 60)
    
    # ========== 1. SÉPARATION DES DONNÉES ==========
    print("\n📊 Étape 1 : Séparation des données")
    print(f"   - Ensemble d'entraînement : {int((1-validation_split)*100)}%")
    print(f"   - Ensemble de validation : {int(validation_split*100)}%")
    
    train_gen, val_gen, class_names = create_data_generators(
        data_dir=data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        validation_split=validation_split
    )
    
    num_classes = len(class_names)
    print(f"   ✓ {num_classes} classes détectées: {class_names}")
    print(f"   ✓ {train_gen.samples} images d'entraînement")
    print(f"   ✓ {val_gen.samples} images de validation")
    
    # ========== 2. CONSTRUCTION DU MODÈLE ==========
    print("\n🔨 Étape 2 : Construction du modèle")
    model = build_cnn_model(
        input_shape=(224, 224, 3),
        num_classes=num_classes
    )
    
    # ========== 3. OPTIMISATION ==========
    print("\n⚙️  Étape 3 : Configuration de l'optimisation")
    print(f"   - Optimiseur : Adam")
    print(f"   - Taux d'apprentissage : {learning_rate}")
    print(f"   - Fonction de perte : categorical_crossentropy")
    print(f"   - Métriques : accuracy, top_k_categorical_accuracy")
    
    model = compile_model(model, learning_rate=learning_rate)
    
    # Afficher le résumé
    print("\n📋 Résumé du modèle :")
    total_params = model.count_params()
    print(f"   - Paramètres totaux : {total_params:,}")
    
    # ========== 4. PRÉVENTION DU SURAPPRENTISSAGE ==========
    print("\n🛡️  Étape 4 : Prévention du surapprentissage")
    print("   ✓ Augmentation des données (rotation, zoom, retournement)")
    print("   ✓ Dropout dans les couches denses (0.5 et 0.3)")
    print("   ✓ BatchNormalization pour stabiliser l'entraînement")
    print("   ✓ EarlyStopping : arrêt si pas d'amélioration")
    print("   ✓ ReduceLROnPlateau : réduction du learning rate si plateau")
    
    # Créer les callbacks
    callbacks = create_callbacks()
    
    # Ajouter des callbacks supplémentaires
    if save_best_model:
        # Créer le dossier models s'il n'existe pas
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        callbacks.append(
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            )
        )
        print(f"   ✓ ModelCheckpoint : sauvegarde du meilleur modèle dans '{model_save_path}'")
    
    # Logger CSV pour l'historique
    csv_logger = CSVLogger('training_history.csv', append=False)
    callbacks.append(csv_logger)
    print(f"   ✓ CSVLogger : historique sauvegardé dans 'training_history.csv'")
    
    # ========== 5. ENTRAÎNEMENT ==========
    print("\n🚀 Étape 5 : Démarrage de l'entraînement")
    print(f"   - Nombre d'époques : {epochs}")
    print(f"   - Taille des lots : {batch_size}")
    print(f"   - Steps par époque (train) : {len(train_gen)}")
    print(f"   - Steps par époque (validation) : {len(val_gen)}")
    print("\n" + "-" * 60)
    
    # Calculer les steps par époque
    steps_per_epoch = len(train_gen)
    validation_steps = len(val_gen)
    
    # Entraîner le modèle
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    print("-" * 60)
    print("\n✅ Entraînement terminé !")
    
    # Afficher les meilleures performances
    best_val_acc = max(history.history['val_accuracy'])
    best_val_loss = min(history.history['val_loss'])
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    
    print(f"\n📊 Meilleures performances :")
    print(f"   - Meilleure précision de validation : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   - Meilleure perte de validation : {best_val_loss:.4f}")
    print(f"   - Époque : {best_epoch}")
    
    # Vérifier si l'objectif de 66% est atteint
    if best_val_acc >= 0.66:
        print(f"\n🎉 Objectif atteint ! Précision >= 66% ({best_val_acc*100:.2f}%)")
    else:
        print(f"\n⚠️  Objectif non atteint. Précision actuelle : {best_val_acc*100:.2f}% (objectif : 66%)")
    
    return model, history


def plot_training_history(history, save_path='training_curves.png'):
    """
    Visualise l'historique d'entraînement
    
    Args:
        history: Historique retourné par model.fit()
        save_path: Chemin pour sauvegarder le graphique
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique 1 : Précision
    axes[0].plot(history.history['accuracy'], label='Précision (entraînement)', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Précision (validation)', marker='s')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Précision')
    axes[0].set_title('Évolution de la précision')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.66, color='r', linestyle='--', label='Objectif (66%)')
    axes[0].legend()
    
    # Graphique 2 : Perte
    axes[1].plot(history.history['loss'], label='Perte (entraînement)', marker='o')
    axes[1].plot(history.history['val_loss'], label='Perte (validation)', marker='s')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Perte')
    axes[1].set_title('Évolution de la perte')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Graphiques sauvegardés dans '{save_path}'")
    plt.show()


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data"
    EPOCHS = 50
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2  # 20% pour validation (80% entraînement)
    LEARNING_RATE = 0.001
    
    # Entraîner le modèle
    model, history = train_model(
        data_dir=DATA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        learning_rate=LEARNING_RATE,
        save_best_model=True,
        model_save_path="models/best_model.h5"
    )
    
    # Visualiser l'historique
    print("\n📈 Génération des graphiques d'entraînement...")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("✓ Entraînement terminé avec succès !")
    print("=" * 60)
    print("\nFichiers générés :")
    print("  - models/best_model.h5 : Meilleur modèle sauvegardé")
    print("  - training_history.csv : Historique détaillé")
    print("  - training_curves.png : Graphiques de performance")
