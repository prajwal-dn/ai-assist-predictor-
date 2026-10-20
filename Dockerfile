# Fetch the real Git LFS model objects in an isolated build stage.
# The source archive supplied by deployment builders can contain LFS pointer text
# instead of the serialized model binaries, so fetch those objects from the public
# GitHub repository and copy only the resolved model artifacts into the final image.
FROM python:3.10-slim AS model-assets

ARG MODEL_SOURCE_REF=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && git clone --depth 1 --branch "${MODEL_SOURCE_REF}" https://github.com/prajwal-dn/ai-assist-predictor-.git /tmp/model-source \
    && git -C /tmp/model-source lfs pull

# Runtime image: keep the deployment lean and run as an unprivileged user.
FROM python:3.10-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies before copying application source to maximize caching.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Replace Git LFS pointer text with the actual model/scaler/feature files.
COPY --from=model-assets --chown=user /tmp/model-source/models/ /app/models/

# Railway routes the generated public domain to this port.
EXPOSE 7860

CMD ["python", "app.py"]
