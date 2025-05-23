FROM python:3.10.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod -R 755 /app
EXPOSE 5000
RUN useradd -m appuser
USER appuser
CMD gunicorn --workers=3 --bind=0.0.0.0:$PORT app:app

