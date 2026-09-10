# CUDA-enabled base: the ensemble runs 12 forward passes per frame and is not
# viable on CPU. The python:3.11-slim base of the template would silently fall
# back to CPU and time out.
# torch >= 2.5 is REQUIRED, and the reason is two steps removed from the symptom:
# transformers 5.x silently DISABLES its PyTorch integration on older torch
# ("Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.1"), and
# with it gone the backbone mixins rfdetr imports are never exported. The
# container then dies on "cannot import name 'BackboneConfigMixin'", which looks
# like a transformers version problem and is not.
FROM --platform=linux/amd64 pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS example-algorithm-amd64

ENV PYTHONUNBUFFERED=1

# Il filtro post-fusione. Misurato su uno sweep a 96 punti
# (runs/cat1_threshold_sweep/sweep.json): con i default di inference.py
# (0.3 / 3 / 0.55) la diagonale vale 0.3258, con questi 0.3631. La quinta li ha
# persi perche' il suo Dockerfile e' stato copiato dalla SECONDA invece che dalla
# quarta, ed e' costato 0.037 di mAP a gratis.
# NB: run_clevis_selftrain.sh dichiara CONTAINER_MIN_SCORE/MAX_PER_FRAME/
# CROSS_CLASS_IOU ma non li scrive mai qui -- lo stage 10 non tocca il
# Dockerfile. Finche' resta cosi', queste righe vanno rimesse a mano ogni round.
ENV CLEVIS_MIN_SCORE=0
ENV CLEVIS_MAX_PER_FRAME=10
ENV CLEVIS_CROSS_CLASS_IOU=1.0

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

# opencv needs these even in the headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
