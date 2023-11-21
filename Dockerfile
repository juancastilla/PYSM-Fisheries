#Set base image in Python 3.10
FROM python:3.11

#Set working directory
WORKDIR /app

#Install git
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

#Clone repo
RUN git clone https://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@github.com/juancastilla/PYSM-Fisheries.git .

#Install re
RUN pip3 install -r requirements.txt

#Expose Port 8501
EXPOSE 8501

#Run healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

#Run Streamlit application
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
