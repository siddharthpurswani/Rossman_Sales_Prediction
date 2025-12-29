FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt gunicorn
EXPOSE 5000
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000", "-w", "1", "--timeout", "120"]