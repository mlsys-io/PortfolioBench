FROM freqtradeorg/freqtrade:stable

USER root

# 1. Install standard data science libraries from official PyPI
RUN pip install --no-cache-dir scikit-learn joblib pandas

# 2. Install PyTorch (CPU version) separately to keep image small
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

USER ftuser