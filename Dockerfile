#Set base image in Python 3.11
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

# replace RUN pip3 install -r /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install numpy==1.26.4                                                                                                                                                                                 
RUN python3 -m pip install -r requirements.txt  

#Expose Port 8080
EXPOSE 8080

#Run healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

#Run Streamlit application
ENTRYPOINT ["streamlit", "run", "streamlit/Home.py", "--server.port=8080", "--server.address=0.0.0.0"]
