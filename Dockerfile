FROM python:3.7

RUN apt-get update ##[edited]

RUN apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /app/

ENTRYPOINT ["streamlit","run"]

CMD [ "streamlit_app.py" ]



