FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

RUN pip install pipenv
RUN pip install pandas pyarrow
COPY ["Pipfile", "Pipfile.lock", "./"]
RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

ENTRYPOINT ["python3", "starter.py"]