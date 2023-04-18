# use production-optimised TFX
FROM tensorflow/serving
LABEL com.opencontainers.image.authors="Diagnox"
LABEL version="1.0"

# copy entire app
ADD . /models/diagnox
WORKDIR /models/diagnox

# expose /models/diagnox as a volume
VOLUME /models/diagnox

# install all other reqs (OpenCV, Flask, etc)
RUN pip install -r requirements.txt

# expose port 8080 (configured in app.py)
EXPOSE 8080
