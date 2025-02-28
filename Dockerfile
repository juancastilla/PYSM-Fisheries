#Set base image in Python 3.11
FROM python:3.12

#Set working directory
WORKDIR /app

#Install git
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy streamlit folder
COPY streamlit /app/streamlit

# Copy requirements.txt
COPY requirements.txt /app/requirements.txt

# replace RUN pip3 install -r /tmp/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install numpy==1.26.4                                                                                                                                                                               
RUN python3 -m pip install -r requirements.txt  

#Expose Port 8502
EXPOSE 8500

#Run healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8500/_stcore/health

#Run Streamlit application
ENTRYPOINT ["streamlit", "run", "streamlit/Home.py", "--server.port=8500", "--server.address=0.0.0.0"]