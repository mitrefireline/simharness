# Deployment Stage
FROM rayproject/ray:2.3.0-py39-gpu as deploy

# Set the correct environment variables
ENV CUDA_VISIBLE_DEVICES=0 \
    HF_CACHE_DIR=None
# Copy the needed code
WORKDIR /code/
COPY requirements.txt README.md LICENSE test.py .
COPY simharness2/ simharness2/

RUN pip install -r requirements.txt

CMD ["python", "test.py"]
