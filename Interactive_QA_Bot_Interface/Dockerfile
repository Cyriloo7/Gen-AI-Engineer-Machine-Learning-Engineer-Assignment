# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any necessary packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r Interactive_QA_Bot_Interface/requirements.txt  

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "Interactive_QA_Bot_Interface/app.py", "--server.port=8501", "--server.address=0.0.0.0"]  
