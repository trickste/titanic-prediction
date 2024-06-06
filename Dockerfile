FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN python3 -m pip install -r requirements.txt ; rm requirements.txt

COPY ./ /app/

EXPOSE 5000

ENTRYPOINT [ "python3", "app.py" ]