FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Install Streamlit
RUN pip install streamlit

CMD ["streamlit","python", "app.py"]
