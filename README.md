# machine_learning
TP machine learning - Reconnaissance d'animaux avec Python et réseaux de neurones

## 📋 État d'avancement du projet

### ✅ Étapes complétées

- [x] **Étape 1 : Collecte et préparation des données**
  - Structure de dossiers créée pour l'organisation des images par classe
  - Organisation : `/data/tigres/`, `/data/elephants/`, `/data/giraffes/`
  - Le système de labellage par dossier est en place (suffisant pour TensorFlow/Keras)
  - Dossier `/data/extra/` créé pour stocker les images supplémentaires (ignoré automatiquement)

- [x] **Étape 2 : Prétraitement essentiel**
  - ✅ Redimensionnement uniforme des images (224x224 par défaut)
  - ✅ Normalisation des valeurs de pixels (0-1)
  - ✅ Augmentation des données :
    - Rotation (jusqu'à 359°)
    - Zoom (jusqu'à 20%)
    - Retournement horizontal
    - Translation (déplacement horizontal/vertical)
  - ✅ Prétraitement à la volée (en mémoire) via les générateurs
  - Script `preprocessing.py` créé et fonctionnel

### 🔄 En cours

- [ ] **Étape 3 : Construction du modèle CNN**
  - Architecture du réseau de neurones
  - Couches convolutionnelles
  - Couches de pooling
  - Couches denses
  - Sortie softmax

### ⏳ À faire

- [ ] **Étape 4 : Entraînement et validation**
  - Séparation train/validation
  - Choix de l'optimiseur
  - Fonction de perte
  - Prévention du surapprentissage (dropout, validation croisée)

- [ ] **Étape 5 : Évaluation et amélioration**
  - Métriques de performance (précision, rappel, matrice de confusion)
  - Analyse des résultats
  - Optimisation du modèle

- [ ] **Étape 6 : Documentation**
  - Rapport détaillé
  - Analyse des résultats
  - Propositions d'amélioration

---

## 📁 Structure du projet

```
machine_learning/
├── data/
│   ├── tigres/        # Images de tigres
│   ├── elephants/     # Images d'éléphants
│   ├── giraffes/      # Images de girafes
│   └── extra/         # Images supplémentaires (ignoré lors du traitement)
├── preprocessing.py    # Script de prétraitement des images ✅
├── requirements.txt    # Dépendances Python ✅
├── README.md           # Ce fichier
└── TP-Reconnaissance-danimaux-avec-Python-et-reseaux-de-neurones.pdf
```

---

## 🚀 Installation

1. Installer les dépendances:
```bash
pip install -r requirements.txt
```

Les dépendances incluent :
- `tensorflow>=2.10.0` - Framework pour les réseaux de neurones
- `numpy>=1.21.0` - Calculs numériques
- `Pillow>=9.0.0` - Traitement d'images
- `matplotlib>=3.5.0` - Visualisation

---

## 📊 Prétraitement des images

### Fonctionnalités implémentées

Le script `preprocessing.py` contient :

1. **Classe `ImagePreprocessor`**
   - Redimensionnement uniforme
   - Normalisation des pixels

2. **Fonction `create_data_generators()`**
   - Création automatique des générateurs d'entraînement et de validation
   - Application de l'augmentation des données
   - Séparation automatique train/validation (80/20)
   - **Ignoration automatique du dossier `extra/`**

3. **Fonction `visualize_augmentation()`**
   - Visualisation des effets de l'augmentation
   - **Ignoration automatique du dossier `extra/`**

4. **Fonction `preprocess_dataset()`**
   - Prétraitement en lot de toutes les images (optionnel, pour sauvegarde manuelle si nécessaire)
   - **Ignoration automatique du dossier `extra/`**
   - Note: Les images sont prétraitées à la volée par les générateurs (pas de sauvegarde par défaut)

### Utilisation

```python
from preprocessing import create_data_generators

# Créer les générateurs de données avec augmentation
train_gen, val_gen, classes = create_data_generators(
    data_dir="data",
    target_size=(224, 224),
    batch_size=32,
    validation_split=0.2
)

# Les générateurs sont prêts pour l'entraînement du modèle CNN
print(f"Classes détectées: {classes}")
print(f"Images d'entraînement: {train_gen.samples}")
print(f"Images de validation: {val_gen.samples}")
```

### Paramètres d'augmentation configurés

- **Rotation** : ±20 degrés
- **Zoom** : ±20%
- **Retournement horizontal** : Activé
- **Translation** : ±10% horizontal et vertical
- **Normalisation** : Valeurs entre 0 et 1

---

## 📝 Notes

- **Objectif de précision** : Le modèle doit atteindre au moins 66% de précision pour être validé
- **Classes** : 3 classes (tigres, éléphants, girafes)
- **Format des images** : JPG, PNG, JPEG acceptés
- **Taille standard** : 224x224 pixels (compatible avec les modèles pré-entraînés)

---

## 🔄 Dernière mise à jour

- ✅ Prétraitement des images implémenté (redimensionnement, normalisation, augmentation)
- ✅ Structure du projet organisée
- ✅ Documentation initiale créée
- ✅ Dossier `extra/` ignoré automatiquement lors du traitement
- ✅ Prétraitement à la volée (pas de sauvegarde, traitement en mémoire pendant l'entraînement)