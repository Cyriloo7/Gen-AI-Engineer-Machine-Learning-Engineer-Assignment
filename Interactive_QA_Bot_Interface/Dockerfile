# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app/Interactive_QA_Bot_Interface

# Copy the contents of the Interactive_QA_Bot_Interface directory into the container
COPY Interactive_QA_Bot_Interface/ .

# Install any necessary packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variable for COHERE_API (default to an empty string if not provided)
ENV COHERE_API=${COHERE_API}

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
