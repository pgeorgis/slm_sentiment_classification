FROM alpine

RUN apk add --no-cache bash \
    python3 python3-dev py3-pip py3-virtualenv \
    build-base linux-headers git gfortran openblas-dev
