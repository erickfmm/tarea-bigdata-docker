FROM python:3
WORKDIR /usr/src/app

RUN apt-get install wget

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

COPY train_wine_model.py .
CMD [ "python", "./train_wine_model.py" ]