FROM python:3.11.14-trixie

WORKDIR ${WORKDIR}

COPY ./requirements.txt .

RUN pip install --no-cache -r requirements.txt

RUN mkdir -p /src

COPY . ${WORKDIR}/.

