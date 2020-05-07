cd docker
docker build . -t nlp-recipes

docker run --gpus all -p 8889:8888 -v ~/notebooks:/root/nlp-recipes-master/examples/text_classification_french --ipc=host nlp-recipes
