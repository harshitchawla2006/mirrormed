FROM python:3.10-slim

# Create user with ID 1000 to resolve Hugging Face Spaces permissions
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user api.py .
COPY --chown=user inference.py .
COPY --chown=user best_model.pt .
COPY --chown=user class_map.json .

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]