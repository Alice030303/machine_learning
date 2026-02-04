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

### ✅ Étapes complétées (suite)

- [x] **Étape 3 : Construction du modèle CNN**
  - ✅ Architecture du réseau de neurones complète
  - ✅ 4 blocs de couches convolutionnelles (32, 64, 128, 256 filtres)
  - ✅ Couches de pooling (MaxPooling2D) pour réduction dimensionnelle
  - ✅ Couches denses (512, 256 neurones) avec Dropout
  - ✅ Sortie softmax pour probabilités par classe
  - ✅ BatchNormalization pour stabiliser l'entraînement
  - ✅ Callbacks (EarlyStopping, ReduceLROnPlateau)
  - Script `model.py` créé et fonctionnel

### ✅ Étapes complétées (suite)

- [x] **Étape 4 : Entraînement et validation**
  - ✅ Séparation train/validation (80/20)
  - ✅ Optimiseur Adam avec learning rate adaptatif
  - ✅ Fonction de perte : categorical_crossentropy
  - ✅ Prévention du surapprentissage :
    - Dropout dans les couches denses
    - Augmentation des données
    - BatchNormalization
    - EarlyStopping
    - ReduceLROnPlateau
  - ✅ Sauvegarde du meilleur modèle
  - ✅ Visualisation des courbes d'entraînement
  - Script `train.py` créé et fonctionnel

### 🔄 En cours

### ⏳ À faire

- [ ] **Étape 5 : Évaluation et amélioration**
  - Métriques de performance (précision, rappel, matrice de confusion)
  - Analyse des résultats
  - Optimisation du modèle

---

## 📁 Structure du projet

```
machine_learning/
├── data/
│   ├── tigre/         # Images de tigres
│   ├── elephant/      # Images d'éléphants
│   ├── giraffe/       # Images de girafes
│   └── extra/         # Images supplémentaires (ignoré lors du traitement)
├── models/            # Modèles sauvegardés (créé après entraînement)
│   └── best_model.h5  # Meilleur modèle sauvegardé
├── preprocessing.py   # Script de prétraitement des images ✅
├── model.py           # Script de construction du modèle CNN ✅
├── train.py           # Script d'entraînement et validation ✅
├── visualize.py       # Script de visualisation de l'augmentation ✅
├── requirements.txt   # Dépendances Python ✅
├── README.md          # Ce fichier
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
   - Génère une image `augmentation_examples.png` avec des exemples

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

### Visualisation de l'augmentation

Pour visualiser les effets de l'augmentation des données, utilisez le script `visualize.py` :

```bash
python visualize.py
```

Ou directement en Python :

```python
from preprocessing import visualize_augmentation

visualize_augmentation(
    data_dir="data",
    target_size=(224, 224),
    num_samples=4  # Nombre d'exemples à afficher
)
```

Cela affichera une fenêtre avec des exemples d'images originales et augmentées, et sauvegardera `augmentation_examples.png`.

### Paramètres d'augmentation configurés

- **Rotation** : ±359 degrés
- **Zoom** : ±20%
- **Retournement horizontal** : Activé
- **Translation** : ±10% horizontal et vertical
- **Normalisation** : Valeurs entre 0 et 1

---

## 🧠 Construction du modèle CNN

### Architecture implémentée

Le script `model.py` contient un modèle CNN complet avec :

1. **4 Blocs de couches convolutionnelles**
   - Bloc 1 : 32 filtres (3x3) + BatchNormalization + MaxPooling
   - Bloc 2 : 64 filtres (3x3) + BatchNormalization + MaxPooling
   - Bloc 3 : 128 filtres (3x3) + BatchNormalization + MaxPooling
   - Bloc 4 : 256 filtres (3x3) + BatchNormalization + MaxPooling

2. **Couches denses**
   - Dense 1 : 512 neurones + Dropout (0.5)
   - Dense 2 : 256 neurones + Dropout (0.3)
   - Sortie : 3 neurones avec activation softmax

3. **Fonctionnalités**
   - BatchNormalization pour stabiliser l'entraînement
   - Dropout pour prévenir le surapprentissage
   - Optimiseur Adam avec learning rate adaptatif
   - Callbacks : EarlyStopping et ReduceLROnPlateau

### Utilisation

```python
from model import build_cnn_model, compile_model, create_callbacks
from preprocessing import create_data_generators

# Créer les générateurs de données
train_gen, val_gen, classes = create_data_generators("data")

# Construire le modèle
model = build_cnn_model(
    input_shape=(224, 224, 3),
    num_classes=len(classes)
)

# Compiler le modèle
model = compile_model(model, learning_rate=0.001)

# Afficher le résumé
model.summary()
```

### Exécution du script

```bash
python model.py
```

Cela affichera :
- L'architecture complète du modèle
- Le nombre de paramètres
- Les callbacks configurés
- Un graphique de l'architecture (si graphviz installé)

---

## 🚂 Entraînement et validation du modèle

### Fonctionnalités implémentées

Le script `train.py` gère l'entraînement complet avec :

1. **Séparation des données**
   - 80% pour l'entraînement
   - 20% pour la validation
   - Séparation automatique via `validation_split`

2. **Optimisation**
   - **Optimiseur** : Adam (Adaptive Moment Estimation)
   - **Learning rate** : 0.001 (adaptatif avec ReduceLROnPlateau)
   - **Fonction de perte** : categorical_crossentropy (entropie croisée)
   - **Métriques** : accuracy, top_k_categorical_accuracy

3. **Prévention du surapprentissage**
   - ✅ **Augmentation des données** : rotation, zoom, retournement
   - ✅ **Dropout** : 0.5 et 0.3 dans les couches denses
   - ✅ **BatchNormalization** : normalisation après chaque couche conv
   - ✅ **EarlyStopping** : arrêt si pas d'amélioration pendant 10 époques
   - ✅ **ReduceLROnPlateau** : réduction du learning rate si plateau

4. **Sauvegarde et suivi**
   - Sauvegarde automatique du meilleur modèle (`models/best_model.h5`)
   - Historique CSV (`training_history.csv`)
   - Graphiques de performance (`training_curves.png`)

### Utilisation

```bash
python train.py
```

Ou en Python :

```python
from train import train_model, plot_training_history

# Entraîner le modèle
model, history = train_model(
    data_dir="data",
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    learning_rate=0.001
)

# Visualiser les résultats
plot_training_history(history)
```

### Paramètres d'entraînement

- **Époques** : 50 (avec arrêt anticipé si nécessaire)
- **Batch size** : 32
- **Validation split** : 0.2 (20%)
- **Learning rate** : 0.001 (adaptatif)

### Fichiers générés

Après l'entraînement, vous obtiendrez :
- `models/best_model.h5` : Meilleur modèle sauvegardé
- `training_history.csv` : Historique détaillé de chaque époque
- `training_curves.png` : Graphiques de précision et perte

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
- ✅ Script `visualize.py` créé pour visualiser l'augmentation des données
- ✅ Correction de la fonction `visualize_augmentation()` (compatibilité Python 3)
- ✅ Modèle CNN construit avec 4 blocs convolutionnels, couches denses, et callbacks
- ✅ **Script d'entraînement complet** avec séparation des données, optimisation, et prévention du surapprentissage