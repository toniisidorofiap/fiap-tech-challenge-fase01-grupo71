import os
import kagglehub
from pathlib import Path
from keras.src.legacy.preprocessing.image import (
    ImageDataGenerator as ImageDataGenerator,
)
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Bibliotecas para Interpretação (Grad-CAM)


# Download latest version
path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")

print("Path to dataset files:", path)


# Define source path (downloaded dataset path)
source_path = Path(path) / "TB_Chest_Radiography_Database"

# Define destination directory inside the project
# Since __file__ is not defined in some environments (e.g., Jupyter notebooks), fallback to current working directory
try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path(os.getcwd())

destination_dir = base_dir / "dataset"

# Create destination directory if it doesn't exist
destination_dir.mkdir(parents=True, exist_ok=True)

# Move the downloaded dataset file to the destination directory
destination_file = destination_dir / source_path.name

# shutil.copytree(str(source_path), str(destination_file))

print(f"Moved dataset to: {destination_file}")

normal_dir = f"{source_path}/Normal"
tuberculosis_dir = f"{source_path}/Tuberculosis"

print(f"Contents of {normal_dir}:")
print(os.listdir(normal_dir)[:10]) # Print first 10 files to avoid long output
print(f"\nNumber of files in {normal_dir}: {len(os.listdir(normal_dir))}")

print(f"\nContents of {tuberculosis_dir}:")
print(os.listdir(tuberculosis_dir)[:10]) # Print first 10 files to avoid long output
print(f"\nNumber of files in {tuberculosis_dir}: {len(os.listdir(tuberculosis_dir))}")

"""## Pré-processamento das imagens

### Subtask:
Realizar o pré-processamento das imagens, incluindo redimensionamento, normalização e possivelmente aumento de dados (data augmentation) para preparar o dataset para a CNN.

**Reasoning**:
Import necessary libraries and define parameters for image preprocessing.
"""

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

"""**Reasoning**:
Create ImageDataGenerators for training and validation sets, including data augmentation for training.


"""

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
    validation_split=0.2 # Split data for validation
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 # Use the same split as training
)

"""**Reasoning**:
Use the ImageDataGenerators to load images from the directories, applying preprocessing and creating data generators for training and validation.


"""

data_dir = source_path

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False # No need to shuffle validation data
)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

"""**Reasoning**:
Flatten the output of the convolutional layers and add dense layers for classification, then print the model summary.


"""

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

"""## Separar os dados

### Subtask:
Dividir o dataset de imagens em conjuntos de treino, validação e teste.

**Reasoning**:
Create the test data generator using ImageDataGenerator, ensuring it uses the 'validation' subset and shuffle is set to False.
"""

test_datagen = ImageDataGenerator(rescale=1./255) # Only rescaling needed for test data

test_generator = test_datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation', # Use the validation subset for testing as per the instructions
    shuffle=False
)


# """**Reasoning**:
# Print the number of images and classes found for the test set.


# """

print(f"Found {test_generator.samples} test images belonging to {test_generator.num_classes} classes.")

# """## Treinar o modelo cnn

# ### Subtask:
# Compilar e treinar o modelo CNN utilizando os dados de treino e validação.

# **Reasoning**:
# Compile the previously defined CNN model with the specified optimizer, loss function, and metrics, then train it using the training and validation data generators for a fixed number of epochs, storing the training history.
# """

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=15, # Training for 15 epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(validation_generator)

print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
