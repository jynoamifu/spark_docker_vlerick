FROM python: 3.8.13

# Install dependencies: do this first
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get install -y ant && \
    apt-get clean

# Create non-root user
RUN useradd -ms/bin/bash python-api
USER python-api 

# Create the source file into the container
WORKDIR /repo 
COPY . /repo 

# Install pip requirements
RUN python -m pip install -r requirements.txt
RUN apt-get update && apt-get install -y spark 

# Define command to run 
EXPOSE 8080
ENTRYPOINT ["python"]
CMD ["ML2.py"]

