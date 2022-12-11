FROM rasa/rasa:latest
COPY app /app
COPY server.sh /app/server.sh


USER root

EXPOSE 5055
#RUN rasa train 
USER root

COPY requirements.txt .

RUN pip install -r requirements.txt
# RUN python -m spacy download en_core_web_md
# WORKDIR /app
# RUN rasa run actions
# CMD [ "rasa ","run","actions" ]
ENTRYPOINT ["/app/server.sh"]
