"""
===================================================================================
PROJETO APS - CIÊNCIA DA COMPUTAÇÃO (5º/6º SEMESTRE) - UNIP
MÓDULO DE MODELAGEM INTELIGENTE: REDE NEURAL CONVOLUCIONAL (VigIA)
===================================================================================
Treinamento supervisionado de mini-CNN para confirmação de presença humana.
Arquitetura convolucional hierárquica baseada em convoluções 2D, max-pooling,
planificação, camadas densas com regularização por abandono (Dropout) e ativação
final sigmoide logística com entropia cruzada binária (Binary Cross-Entropy).
===================================================================================
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image


def preparar_dataset_local(diretorio_base="dataset", diretorio_reais="imagens_reais"):
    """
    Garante a estrutura de diretórios e amostras balanceadas para treino e validação.
    """
    pasta_vazio = os.path.join(diretorio_base, "vazio")
    pasta_ocupado = os.path.join(diretorio_base, "ocupado")
    os.makedirs(pasta_vazio, exist_ok=True)
    os.makedirs(pasta_ocupado, exist_ok=True)

    if len(os.listdir(pasta_vazio)) > 10 and len(os.listdir(pasta_ocupado)) > 10:
        return

    caminho_vazio = os.path.join(diretorio_reais, "escritorio_vazio.jpg")
    caminho_ocupado = os.path.join(diretorio_reais, "escritorio_com_pessoas.jpg")

    if not os.path.exists(caminho_vazio) or not os.path.exists(caminho_ocupado):
        return

    img_vazio = Image.open(caminho_vazio).convert("RGB")
    img_ocupado = Image.open(caminho_ocupado).convert("RGB")

    idx = 0
    for w in [200, 250, 300, 350]:
        for x in range(0, img_vazio.width - w + 1, 60):
            for y in range(0, img_vazio.height - w + 1, 50):
                crop = img_vazio.crop((x, y, x + w, y + w)).resize((64, 64))
                crop.save(os.path.join(pasta_vazio, f"vazio_{idx:03d}.jpg"))
                idx += 1
                crop.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(pasta_vazio, f"vazio_{idx:03d}.jpg"))
                idx += 1
    img_vazio.resize((64, 64)).save(os.path.join(pasta_vazio, f"vazio_{idx:03d}.jpg"))
    idx += 1
    img_vazio.resize((64, 64)).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(pasta_vazio, f"vazio_{idx:03d}.jpg"))

    idx = 0
    for w in [180, 220, 260, 300]:
        for x in range(0, img_ocupado.width - w + 1, 50):
            for y in range(0, img_ocupado.height - w + 1, 40):
                crop = img_ocupado.crop((x, y, x + w, y + w)).resize((64, 64))
                crop.save(os.path.join(pasta_ocupado, f"ocupado_{idx:03d}.jpg"))
                idx += 1
                crop.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(pasta_ocupado, f"ocupado_{idx:03d}.jpg"))
                idx += 1
    img_ocupado.resize((64, 64)).save(os.path.join(pasta_ocupado, f"ocupado_{idx:03d}.jpg"))
    idx += 1
    img_ocupado.resize((64, 64)).transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(pasta_ocupado, f"ocupado_{idx:03d}.jpg"))


def treinar_modelo_vigia(caminho_dataset="dataset", caminho_saida="modelo_vigia.keras", epocas=10):
    """
    Treinamento conciso da mini-CNN (~15 linhas Keras essencial).
    """
    preparar_dataset_local(caminho_dataset)
    dataset = tf.keras.utils.image_dataset_from_directory(
        caminho_dataset, image_size=(64, 64), batch_size=8, class_names=["vazio", "ocupado"]
    )
    modelo = models.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    modelo.fit(dataset, epochs=epocas, verbose=1)
    modelo.save(caminho_saida)
    return modelo


if __name__ == "__main__":
    treinar_modelo_vigia()
