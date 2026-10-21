FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY student_result_model.pkl .

EXPOSE 5000

CMD ["python", "app.py"]
