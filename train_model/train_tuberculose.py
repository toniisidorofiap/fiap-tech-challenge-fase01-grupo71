# -*- coding: utf-8 -*-
"""tuberculosis_project.py - Script Unificado"""

import os
from pathlib import Path
import shutil
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import (
    ImageDataGenerator as ImageDataGenerator,
)
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub

# Bibliotecas para Interpretação (Grad-CAM)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore

# --- 1. CONFIGURAÇÕES GLOBAIS ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
DATASET_NAME = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
BASE_DIR = base_dir = Path(__file__).parent / 'dataset' # Pode ser ajustado para um caminho local se necessário

# --- 2. DOWNLOAD E ORGANIZAÇÃO DOS DADOS (KAGGLE) ---

print("1. Baixando e organizando o dataset do Kaggle...")

try:
    # Baixa o dataset
    kaggle_path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset baixado para: {kaggle_path}")

    # Definindo caminhos de origem e destino
    source_normal = os.path.join(kaggle_path, 'TB_Chest_Radiography_Database', 'Normal')
    source_tuberculosis = os.path.join(kaggle_path, 'TB_Chest_Radiography_Database', 'Tuberculosis')
    
    # Se estiver rodando no Windows/local, você pode querer mudar o BASE_DIR
    # para um caminho local C:\...\data
    if os.name == 'nt': 
        BASE_DIR = os.path.join(os.getcwd(), 'local_data')
    
    dest_normal = os.path.join(BASE_DIR, 'Normal')
    dest_tuberculosis = os.path.join(BASE_DIR, 'Tuberculosis')
    
    if os.path.exists(BASE_DIR):
        shutil.rmtree(BASE_DIR)
        
    os.makedirs(dest_normal, exist_ok=True)
    os.makedirs(dest_tuberculosis, exist_ok=True)

    def copy_files(source_dir, dest_dir):
        count = 0
        if not os.path.exists(source_dir):
             print(f"AVISO: Diretório de origem não encontrado: {source_dir}")
             return 0
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(source_dir, filename), dest_dir)
                count += 1
        return count

    num_normal = copy_files(source_normal, dest_normal)
    num_tuberculosis = copy_files(source_tuberculosis, dest_tuberculosis)

    print(f"Número de imagens 'Normal' copiadas: {num_normal}")
    print(f"Número de imagens 'Tuberculosis' copiadas: {num_tuberculosis}")

except Exception as e:
    print(f"Erro ao baixar ou organizar os dados do Kaggle: {e}")
    # Se houver erro, para a execução, pois os dados são essenciais
    exit() # Interrompe a execução do script

# --- 3. PRÉ-PROCESSAMENTO E CRIAÇÃO DE GENERATORS ---

print("\n2. Configurando pré-processamento e data generators...")

# Configuração para Treinamento (com data augmentation e normalização)
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalização
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2,
    height_shift_range=0.2,
    width_shift_range=0.2,
    validation_split=0.2 # 80% para treino
)

# Configuração para Validação/Teste (somente normalização)
val_test_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Gerador de Treinamento
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Gerador de Validação/Teste
# Este será usado tanto para a validação no treino quanto para a avaliação final
validation_generator = val_test_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Imagens de Treino encontradas: {train_generator.samples}")
print(f"Imagens de Validação/Teste encontradas: {validation_generator.samples}")

# --- 4. CONSTRUÇÃO DO MODELO CNN ---

print("\n3. Construindo o modelo CNN...")

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', name='conv2d_2'), # Nomeando a última camada conv para Grad-CAM
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

# --- 5. TREINAMENTO E AVALIAÇÃO DO MODELO ---

print("\n4. Compilando e treinando o modelo...")

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

EPOCHS = 15
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

print("\n5. Avaliando o modelo...")

loss, accuracy = model.evaluate(validation_generator)
print(f"\nPerda Final (Teste): {loss:.4f}")
print(f"Acurácia Final (Teste): {accuracy:.4f}")

# Previsões para relatório de classificação
validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = np.round(Y_pred).flatten()
y_true = validation_generator.classes

print("\nRelatório de Classificação (Conjunto de Teste):")
target_names = list(train_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

print("\nMatriz de Confusão:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# --- 6. INTERPRETAÇÃO DO MODELO (GRAD-CAM) ---

print("\n6. Realizando a interpretação do modelo com Grad-CAM...")

class BinaryScore(CategoricalScore):
    def __call__(self, output):
        # Queremos a ativação máxima para a classe positiva (Tuberculosis, que é o índice 1)
        # O modelo outputa a probabilidade da classe 1. 
        # Esta função faz o Grad-CAM focar na classe predita.
        return output[:, 0]

# --- 6.1. Preparação para a Visualização ---
LAST_CONV_LAYER = 'conv2d_2' # Última camada Conv2D nomeada
validation_generator.reset()
# Pega um batch de dados de teste (normalizado)
X_test, Y_test = validation_generator.next() 

gradcam = Gradcam(
    model,
    model_modifier=ReplaceToLinear(), 
    clone=True
)

# Encontra índices preditos como Tuberculose (1)
predicted_classes = np.round(model.predict(X_test)).flatten()
predicted_tuberculosis_indices = np.where(predicted_classes == 1)[0]

if len(predicted_tuberculosis_indices) == 0:
    print("\nAVISO: Não foi possível encontrar imagens preditas como Tuberculose no primeiro batch. Pulando a visualização Grad-CAM.")
else:
    indices_to_show = predicted_tuberculosis_indices[:min(4, len(predicted_tuberculosis_indices))]

    print(f"\nGerando Grad-CAM para {len(indices_to_show)} imagens preditas como 'Tuberculosis'...")
    fig, axes = plt.subplots(len(indices_to_show), 3, figsize=(10, 3 * len(indices_to_show)))
    if len(indices_to_show) == 1:
        axes = [axes] # Envolve em uma lista para iterar

    for i, idx in enumerate(indices_to_show):
        img = X_test[idx][np.newaxis, ...]
        true_label = target_names[int(Y_test[idx])]
        predicted_label = target_names[int(predicted_classes[idx])]

        # Gera o mapa de ativação
        cam = gradcam(BinaryScore(), img, penultimate_layer=LAST_CONV_LAYER)
        cam = cam[0, :]
        
        ax_orig, ax_heat, ax_overlay = axes[i] if len(indices_to_show) > 1 else axes
        
        # Plotagem da Imagem Original
        ax_orig.imshow(img[0])
        ax_orig.set_title(f"Original (True: {true_label})")
        ax_orig.axis('off')

        # Plotagem da Sobreposição (Original + Heatmap)
        
        # Redimensiona o CAM para o tamanho da imagem e aplica o heatmap
        cam_resized = tf.image.resize(cam[..., np.newaxis], (IMG_HEIGHT, IMG_WIDTH), method='bilinear')[..., 0]
        cam_resized = (cam_resized.numpy() - cam_resized.numpy().min()) / (cam_resized.numpy().max() - cam_resized.numpy().min())
        
        # Mapeia cores do CAM
        heatmap_overlay = cm.jet(cam_resized)[:, :, :3]

        # Converte a imagem de volta para 0-255, depois normaliza para a mistura
        superimposed_img = np.uint8(img[0] * 255)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        superimposed_img = tf.keras.preprocessing.image.img_to_array(superimposed_img)
        
        # Combina imagem original e heatmap
        superimposed_img = (superimposed_img / 255) * 0.4 + heatmap_overlay * 0.6 
        
        ax_overlay.imshow(superimposed_img)
        ax_overlay.set_title(f"Overlay (Pred: {predicted_label})")
        ax_overlay.axis('off')
        
        # Plotagem do Heatmap puro (opcional)
        ax_heat.imshow(heatmap_overlay)
        ax_heat.set_title("Heatmap (CAM)")
        ax_heat.axis('off')


    plt.tight_layout()
    plt.show()

# --- 7. DISCUSSÃO CRÍTICA (Texto) ---

print("\n\n--- 7. DISCUSSÃO CRÍTICA E APLICABILIDADE PRÁTICA ---")
print("\n1. Desempenho e Viés do Modelo:")
print("O alto Recall é crucial para o diagnóstico de Tuberculose, pois minimiza Falsos Negativos (casos de TB perdidos). O modelo deve ser validado contra o viés do dataset, que pode não refletir a baixa prevalência real da doença.")
print("\n2. Interpretabilidade (Grad-CAM):")
print("O Grad-CAM fornece a 'prova' visual de que o modelo está focando nas regiões pulmonares clinicamente relevantes (consolidações, cavitações). Se o foco cair em artefatos ou bordas da imagem, o modelo é frágil e inconfiável.")
print("\n3. O Papel Final do Médico:")
print("A IA é uma ferramenta de apoio (auxiliar de triagem), nunca de substituição. O médico é o único profissional capaz de integrar a informação da imagem com o histórico clínico, sintomas e outros exames (ex: cultura), mantendo a responsabilidade e o discernimento clínico no diagnóstico final.")