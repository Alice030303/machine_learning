import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
import random
import pandas as pd
import numpy as np
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from keras import optimizers
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import load_img, img_to_array

elephant_dir = './data/elephant'
tigre_dir = './data/tigre'
giraffe_dir = './data/giraffe'




elephant = os.listdir(elephant_dir)
tigre = os.listdir(tigre_dir)
giraffe = os.listdir(giraffe_dir)

df=pd.DataFrame({'filename': elephant + tigre + giraffe,
                 'label': ['elephant'] * len(elephant) + ['tigre'] * len(tigre) + ['giraffe'] * len(giraffe),
                 })

print(df.head())

elephant_df = pd.DataFrame({
    'filename': ['elephant/' + f for f in elephant],
    'label': 'elephant'
}).sample(n=800, random_state=42)

tigre_df = pd.DataFrame({

    'filename': ['tigre/' + f for f in tigre],
    'label': 'tigre'
}).sample(n=800, random_state=42)

giraffe_df = pd.DataFrame({

    'filename': ['giraffe/' + f for f in giraffe],
    'label': 'giraffe'
}).sample(n=800, random_state=42)

df_final = pd.concat([elephant_df, tigre_df, giraffe_df]).reset_index(drop=True)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
df_final['label'] = df_final['label'].astype(str)


current_dir = os.getcwd()
data_path = os.path.join(current_dir, "data")

train_generator = train_datagen.flow_from_dataframe(
    df_final,
    directory=data_path, 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_dataframe(
    dataframe=df_final,
    directory=data_path, 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = models.Sequential()


model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # Couche de convolution : 32 filtres de 3x3, détecte des motifs simples dans l'image, ReLU pour rendre le modèle non-linéaire, forme d'entrée (150x150x3).

model.add(layers.MaxPooling2D((2, 2)))  # Couche de max pooling : réduit les dimensions de l'image en prenant le maximum de chaque carré de 2x2 pixels.

model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Deuxième couche de convolution : 64 filtres de 3x3, détecte des motifs plus complexes, ReLU pour l'activation.

model.add(layers.MaxPooling2D((2, 2)))  # Deuxième couche de max pooling : réduit encore les dimensions de l'image.

model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # Troisième couche de convolution : 128 filtres de 3x3, détecte des motifs encore plus complexes, ReLU pour l'activation.

model.add(layers.MaxPooling2D((2, 2)))  # Troisième couche de max pooling : réduit encore les dimensions de l'image.

model.add(layers.Flatten())  # Couche Flatten : transforme les données 2D en un vecteur 1D pour la couche dense suivante.

model.add(layers.Dense(512, activation='relu'))  # Couche dense (fully connected) : 512 neurones, chaque neurone est connecté à tous les neurones de la couche précédente, ReLU pour l'activation.

model.add(layers.Dense(3, activation='softmax'))



optimi = optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=optimi,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(
    train_generator, 

    steps_per_epoch=train_generator.samples // train_generator.batch_size,

    epochs=5,  

    validation_data=validation_generator,  

    validation_steps=validation_generator.samples // validation_generator.batch_size
)

validation_steps = validation_generator.samples // validation_generator.batch_size
evaluation = model.evaluate(validation_generator, steps=validation_steps)


print(f"Perte sur la validation: {evaluation[0]}")
print(f"Précision sur la validation: {evaluation[1]}")


save_path = './model_animal_classifier.h5'
model.save(save_path)
print(f"Modèle sauvegardé à l'emplacement : {save_path}")



test_dir = './data/extra'
test = os.listdir(test_dir)



model = load_model('/content/drive/My Drive/data/shape_model.h5')

elephant_df = pd.DataFrame({
    'filename': ['elephant/' + f for f in elephant],
    'label': 'elephant'
}).sample(n=10, random_state=42)

tigre_df = pd.DataFrame({
    'filename': ['tigre/' + f for f in tigre],
    'label': 'tigre'
}).sample(n=10, random_state=42)

giraffe_df = pd.DataFrame({
    'filename': ['giraffe/' + f for f in giraffe],
    'label': 'giraffe'
}).sample(n=10, random_state=42)


test_df = pd.DataFrame({
    'filename': test,

}).sample(n=10, random_state=42)


df_final = pd.concat([elephant_df, tigre_df, giraffe_df]).reset_index(drop=True)
df_final
df_final['label'] = df_final['label'].astype(str)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    df_final,
    directory="/content/drive/My Drive/data",
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    batch_size=10,
    class_mode='categorical',
    shuffle=False
)


predict_datagen = ImageDataGenerator(rescale=1./255)
predict_generator = predict_datagen.flow_from_dataframe(
    test_df,
    directory="/data/extra",
    x_col='filename',
    target_size=(150, 150),
    batch_size=10,
    class_mode=None,
    shuffle=False
)


predictions = model.predict(test_generator)


predicted_classes_indices = np.argmax(predictions, axis=1)
class_labels = list(test_generator.class_indices.keys())




predicted_classes = [class_labels[i] for i in predicted_classes_indices]


df_final['predictions'] = predicted_classes

df_final

