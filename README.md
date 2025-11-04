# FIAP - IA PARA DEVS - Tech Challenge Fase 01 - Grupo 71

# Overview
Este projeto fornece uma API para análise de imagens de exames raio-x para detecção de tuberculose, além de uma interface web feita com Streamlit para o upload e visualização dos resultados.

# Requisitos
- Python 3.10+ (para rodar local)
- Docker e Docker Compose (para rodar em containers)

# Treinando o Modelo
Dataset utilizado: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
O modelo pode ser treinado localmente utilizando o notebook `tech_challenge_tuberculose.ipynb` localizado na pasta `train_model/`.
Alternativamente, você pode abrir este notebook diretamente no [Google Colab](https://colab.research.google.com/) para execução na nuvem, sem necessidade de configuração local.

Para treinar o modelo no Mac com processador silicon (M1, M2, M3, M4, etc) recomenda-se usar o conda e instalar o tensorflow-macos e tensorflow-metal habilitando assim o uso de gpu.
```bash
brew install --cask miniforge 
conta init zsh # ou bash se não estiver usando zsh
conda create -p ./.fiap-tf python=3.9
conda activate ./.fiap-tf
python -m pip install --upgrade pip setuptools wheel
pip install tensorflow-macos tensorflow-metal
# Não rode pip install tensorflow (isso instala a versão CPU/Intel e pode conflitar).
pip install -r req_no_tf_mac.txt # da o install no requirements sem o tensorflow declarado
# se der erro no opencv-python tente conda install -c conda-forge opencv

#Verificação rápida — confirmar TF + Metal/GPU, se não der nenhum erro o setup no mac esta correto
python - <<'PY'
import time, numpy as np, tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Physical devices:", tf.config.list_physical_devices())
gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", gpus)
# Quick matmul to exercise device
a = tf.random.uniform([2000,2000])
b = tf.random.uniform([2000,2000])
t0 = time.time()
c = tf.matmul(a,b)
# force execution
_ = c.numpy()
print("Matmul time (s):", time.time() - t0)
PY
```


# Como rodar o projeto
## Rodando localmente (sem Docker)
1. Crie e ative um ambiente virtual com Python 3.10+:
	```bash
	python3 -m venv venv
	source venv/bin/activate  # Linux/macOS
	venv\Scripts\activate   # Windows
	```
2. Instale as dependências:
	```bash
	pip install -r requirements.txt
	```
3. Inicie a API:
	```bash
	uvicorn api.main:app --host 0.0.0.0 --port 8888 --reload
	```
4. Inicie a interface web:
	```bash
	streamlit run web/app.py
	```

## Rodando com Docker e Docker Compose
1. Certifique-se de que Docker e Docker Compose estejam instalados.
2. Construa e suba os serviços:
	```bash
	docker-compose up --build
	```
3. A API estará disponível em `http://localhost:8888`.
4. A aplicação web estará disponível em `http://localhost:8501`.

# Notas
- Utilize o campo de upload da web para carregar múltiplas imagens de exames.
- Caso queira limpar resultados, use o botão de limpar.

