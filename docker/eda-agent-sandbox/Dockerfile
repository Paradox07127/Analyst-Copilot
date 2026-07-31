FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HOME=/tmp/home
ENV TMPDIR=/tmp
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CACHE_HOME=/tmp/cache

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock

RUN groupadd --gid 65532 sandbox \
    && useradd --no-create-home --uid 65532 --gid 65532 --shell /usr/sbin/nologin sandbox \
    && mkdir -p /inputs /sandbox /work \
    && chown sandbox:sandbox /work

WORKDIR /work
USER sandbox:sandbox
CMD ["python", "-I", "-B", "-u", "/sandbox/analysis.py"]
