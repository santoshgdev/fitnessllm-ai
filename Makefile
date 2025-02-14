clean:
	poetry env remove --all
	poetry install --sync

build:
	docker build . -t fitnessllm-ai

test:
	poetry run pytest --cov --cov-branch --cov-report=html

run:
	docker run -it -v ${CODE_PATH}/fitnessllm-dataplatform:/app/fitnessllm-ai \
				   -v ~/.config/gcloud:/root/.config/gcloud \
				   fitnessllm-ai:latest \
				   zsh

check_gpu:
	docker run --gpus all fitnessllm-ai nvidia-smi