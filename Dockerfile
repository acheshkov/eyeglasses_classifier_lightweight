FROM pytorch/pytorch:latest
RUN apt-get update
RUN pip install pandas sklearn
RUN apt-get install -y zip vim
COPY ./dataset.zip .
COPY ./*.py ./

# ENTRYPOINT ["python"]
# CMD ["inference.py" "--images-path='./images'" "--model-params-path='./dist/squeezenet_params'"]

CMD [ "python", "./main.py" ]