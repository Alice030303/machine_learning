"""
Construction du modèle CNN pour la reconnaissance d'animaux
- Couches convolutionnelles pour l'extraction de caractéristiques
- Couches de pooling pour la réduction de dimensionnalité
- Couches denses pour la classification
- Sortie softmax pour les probabilités
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from keras.metrics import TopKCategoricalAccuracy


def build_cnn_model(input_shape=(224, 224, 3), num_classes=3):
    """
    Construit un modèle CNN pour la classification d'images
    
    Args:
        input_shape: Taille des images d'entrée (height, width, channels)
        num_classes: Nombre de classes à classifier
        
    Returns:
        Modèle CNN compilé
    """
    model = Sequential([
        # ========== BLOC 1 : Extraction de caractéristiques de base ==========
        # Couche convolutionnelle 1
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='conv1'),
        BatchNormalization(name='bn1'),  # Normalisation pour stabiliser l'entraînement
        MaxPooling2D(2, 2, name='pool1'),
        
        # ========== BLOC 2 : Caractéristiques plus complexes ==========
        # Couche convolutionnelle 2
        Conv2D(64, (3, 3), activation='relu', name='conv2'),
        BatchNormalization(name='bn2'),
        MaxPooling2D(2, 2, name='pool2'),
        
        # ========== BLOC 3 : Caractéristiques avancées ==========
        # Couche convolutionnelle 3
        Conv2D(128, (3, 3), activation='relu', name='conv3'),
        BatchNormalization(name='bn3'),
        MaxPooling2D(2, 2, name='pool3'),
        
        # ========== BLOC 4 : Caractéristiques très complexes ==========
        # Couche convolutionnelle 4
        Conv2D(256, (3, 3), activation='relu', name='conv4'),
        BatchNormalization(name='bn4'),
        MaxPooling2D(2, 2, name='pool4'),
        
        # ========== TRANSITION : Aplatissement ==========
        Flatten(name='flatten'),
        
        # ========== COUCHES DENSES : Classification ==========
        # Couche dense 1
        Dense(512, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout1'),  # Prévention du surapprentissage
        
        # Couche dense 2
        Dense(256, activation='relu', name='dense2'),
        Dropout(0.3, name='dropout2'),
        
        # ========== SORTIE : Probabilités pour chaque classe ==========
        Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile le modèle avec optimiseur et fonction de perte
    
    Args:
        model: Modèle CNN à compiler
        learning_rate: Taux d'apprentissage
        
    Returns:
        Modèle compilé
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy', 
        metrics=['accuracy', TopKCategoricalAccuracy(k=2, name='top_2_acc')]
    )
    
    return model


def print_model_summary(model):
    """
    Affiche un résumé détaillé du modèle
    
    Args:
        model: Modèle CNN
    """
    print("\n" + "=" * 60)
    print("ARCHITECTURE DU MODÈLE CNN")
    print("=" * 60)
    model.summary()
    
    # Compter le nombre de paramètres
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"\n📊 Statistiques du modèle:")
    print(f"  - Paramètres totaux: {total_params:,}")
    print(f"  - Paramètres entraînables: {trainable_params:,}")
    print(f"  - Paramètres non-entraînables: {total_params - trainable_params:,}")


def visualize_model_architecture(model, filename='model_architecture.png'):
    """
    Visualise l'architecture du modèle
    
    Args:
        model: Modèle CNN
        filename: Nom du fichier de sortie
    """
    try:
        keras.utils.plot_model(
            model,
            to_file=filename,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB'
        )
        print(f"\n✓ Architecture du modèle sauvegardée dans '{filename}'")
    except Exception as e:
        print(f"\n⚠ Impossible de générer le graphique: {e}")
        print("   Installez graphviz et pydot pour la visualisation:")

def create_callbacks():
    """
    Crée les callbacks pour améliorer l'entraînement
    
    Returns:
        Liste de callbacks
    """
    callbacks = [
        # Arrêt anticipé si la validation ne s'améliore plus
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Attendre 10 époques sans amélioration
            restore_best_weights=True,
            verbose=1
        ),
        
        # Réduction du taux d'apprentissage si plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Réduire de moitié
            patience=5,  # Attendre 5 époques
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks


if __name__ == "__main__":
    # Configuration
    INPUT_SHAPE = (224, 224, 3)  # Images RGB 224x224
    NUM_CLASSES = 3  # elephant, tigre, giraffe
    LEARNING_RATE = 0.001
    
    print("=" * 60)
    print("CONSTRUCTION DU MODÈLE CNN")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - Taille d'entrée: {INPUT_SHAPE}")
    print(f"  - Nombre de classes: {NUM_CLASSES}")
    print(f"  - Taux d'apprentissage: {LEARNING_RATE}")
    
    # Construire le modèle
    print("\n🔨 Construction du modèle...")
    model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    
    # Compiler le modèle
    print("⚙️  Compilation du modèle...")
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    # Afficher le résumé
    print_model_summary(model)
    
    # Visualiser l'architecture (optionnel)
    print("\n📈 Génération du graphique de l'architecture...")
    visualize_model_architecture(model)
    
    # Afficher les callbacks
    callbacks = create_callbacks()
    print(f"\n✅ Callbacks configurés:")
    print(f"  - EarlyStopping: arrêt si pas d'amélioration pendant 10 époques")
    print(f"  - ReduceLROnPlateau: réduction du taux d'apprentissage si plateau")
    
    print("\n" + "=" * 60)
    print("✓ Modèle CNN construit avec succès!")
    print("=" * 60)
    print("\nLe modèle est prêt pour l'entraînement.")
    print("\nPour entraîner le modèle, utilisez:")
    print("  from model import build_cnn_model, compile_model")
    print("  from preprocessing import create_data_generators")
    print("  ")
    print("  # Créer les générateurs")
    print("  train_gen, val_gen, classes = create_data_generators('data')")
    print("  ")
    print("  # Construire et compiler le modèle")
    print("  model = build_cnn_model(num_classes=len(classes))")
    print("  model = compile_model(model)")
    print("  ")
    print("  # Entraîner")
    print("  history = model.fit(train_gen, validation_data=val_gen, epochs=50)")
