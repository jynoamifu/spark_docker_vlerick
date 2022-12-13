FROM python: 3.8.13

# Create non-root user
RUN useradd -ms/bin/bash python-api
USER python-api 

# Create the source file into the container
WORKDIR /repo 
COPY . /repo 

# Install pip requirements
RUN python -m pip install -r requirements.txt
RUN apt-get update && apt-get install -y spark 

# Install dependencies 
RUN 



# RUN ML Code

