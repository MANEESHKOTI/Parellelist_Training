FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
