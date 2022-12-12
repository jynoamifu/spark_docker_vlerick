FROM python: 3.8

# Create non-root user
RUN useradd -ms/bin/bash python-api
USER python-api 


RUN apt-get update && apt-get install -y spark 

RUN pip install pyspark



# Copy source file into a container
WORKDIR 
COPY

# Install pip requirements 
RUN pip install requirements.txt

# RUN ML Code

