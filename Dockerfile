FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV CONFIG_CONNECTION_STRING=Endpoint=https://app-config-predictive.azconfig.io;Id=K39n;Secret=Z0oNxiCBtKfIk90Gj2Yftdlv85XPD76uL1/sGCLwy1k=

CMD ["python","flask_app.py"]
