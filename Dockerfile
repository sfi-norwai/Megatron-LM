FROM nvcr.io/nvidia/pytorch:22.01-py3

WORKDIR /app

COPY . .
RUN chmod +x norgpt3.sh

ENTRYPOINT ["./norgpt3.sh"]
