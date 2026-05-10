# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Setup user for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# FIX: Change port from 8501 to 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
