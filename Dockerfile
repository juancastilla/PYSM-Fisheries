#Set base image in Python 3.10
FROM python:3.10

#Expose Port 8501
EXPOSE 8501

#Set working directory
WORKDIR /app

#Copy packages required from local requirements file to Docker image requirements file
COPY requirements.txt ./requirements.txt

#Install dependencies
RUN pip3 install -r requirements.txt

#Copy all files from local project to Docker image
COPY . .

#Run Streamlit application
CMD streamlit run ./streamlit/Home.py