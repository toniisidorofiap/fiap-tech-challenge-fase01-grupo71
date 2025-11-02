# FIAP - IA PARA DEVS - Tech Challenge Fase 01 - Grupo 71

# Overview
Este projeto fornece uma API para análise de imagens de exames raio-x para detecção de tuberculose, além de uma interface web feita com Streamlit para o upload e visualização dos resultados.

# Requisitos
- Python 3.10+ (para rodar local)
- Docker e Docker Compose (para rodar em containers)

# Como rodar o projeto

# Treinando o Modelo
O modelo pode ser treinado localmente utilizando o notebook `tech_challenge_tuberculose.ipynb` localizado na pasta `train_model/`.
Alternativamente, você pode abrir este notebook diretamente no [Google Colab](https://colab.research.google.com/) para execução na nuvem, sem necessidade de configuração local.

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

