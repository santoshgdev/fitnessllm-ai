clean:
	poetry env remove --all
	poetry install --sync

build:
	DOCKER_BUILDKIT=1 docker build . -t fitnessllm-ai --progress=plain

test:
	poetry run pytest --cov --cov-branch --cov-report=html

run:
	docker run -it -v ${CODE_PATH}/fitnessllm-dataplatform:/app/fitnessllm-ai \
				   -v ~/.config/gcloud:/root/.config/gcloud \
				   fitnessllm-ai:latest \
				   zsh

ollama:
    docker run --gpus all --network fitnessllm_network -e OLLAMA_HOST=0.0.0.0 -p 11434:11434 --name ollama ollama/ollama


check_gpu:
	docker run --gpus all fitnessllm-ai nvidia-smi
