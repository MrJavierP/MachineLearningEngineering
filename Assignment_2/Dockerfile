# Use the official Apache Airflow image (adjust the version as needed)
FROM apache/airflow:2.6.1

# Switch to root to install additional packages
USER root

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
RUN rm -f /etc/apt/sources.list.d/*.list && \
    printf "deb http://archive.debian.org/debian bullseye main\ndeb http://archive.debian.org/debian-security bullseye-security main\n" > /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /bin/bash /bin/sh

# Set JAVA_HOME to the directory expected by Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Switch to the airflow user before installing Python dependencies
USER airflow

# Install Python dependencies using requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create a volume mount point for notebooks
VOLUME /app
