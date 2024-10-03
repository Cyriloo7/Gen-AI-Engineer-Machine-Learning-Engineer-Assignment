# Part 1: [Retrieval-Augmented Generation (RAG) Model for QA Bot](<Retrieval-Augmented Generation (RAG) Model for QA Bot>)

requirements.txt: 
```bash
faiss-cpu
cohere
PyPDF2
numpy
```

Architecture:
![RAG Model Architecture](https://github.com/user-attachments/assets/5efd11a4-56e1-4ada-a07a-daed2d2807ef)


# Part 2: [Interactive QA Bot Interface](<Interactive_QA_Bot_Interface>)

Docker Hub pull command: 
Using docker images from Docker Hub, you can access this application.

Pull the Application from Docker Hub
```bash
docker pull cyriljose/interactive_qa_bot:1.0
```

Run the Application
```bash
docker run -p 8501:8501 cyriljose/interactive_qa_bot:1.0
```

Requirments.txt
```bash
streamlit
pandas
numpy
PyPDF2
cohere
faiss-cpu
```
