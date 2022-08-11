FROM python:3.9-slim

RUN mkdir /opt/mlwb
ADD train.py /opt/mlwb/train.py
WORKDIR /opt/mlwb

RUN pip install --quiet --no-cache-dir torch==1.12.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --quiet --no-cache-dir torch-scatter==2.0.9 torch-sparse==0.6.14 torch-cluster==1.6.0 torch-spline-conv==1.2.1 torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.12.0+cpu.html && \
    pip install --quiet --no-cache-dir \
        class-resolver==0.3.9 \
        pyTigerGraph==1.0.2

RUN chgrp -R 0 /opt/mlwb \
    && chmod -R g+rwX /opt/mlwb

ENTRYPOINT ["python3", "/opt/mlwb/train.py"]