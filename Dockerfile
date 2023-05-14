   FROM python:3.8.10

   WORKDIR /app

   # // USE COPY FOR THE LOCAL //
   # COPY . .
   
   RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

   RUN git clone https://github.com/torchme/mfdp.git .
   
   RUN /usr/local/bin/python -m pip install --upgrade pip
   RUN pip install --no-cache-dir -r requirements.txt

   EXPOSE 8081

   HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
   
   ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8081", "--server.address=0.0.0.0"]
