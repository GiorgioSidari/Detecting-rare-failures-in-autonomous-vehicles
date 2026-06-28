FROM python:3.8.20-bookworm

WORKDIR /simulator

RUN apt-get update && apt-get install -y \
    libgl1 xvfb \
    libglu1-mesa \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libasound2 \
    libpulse0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Add open-sbt requirements
COPY ./Simulator/lanekeeping/requirements.txt /simulator
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Add Simulation server requirements
COPY ./Simulator/requirements.txt /simulator
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy the simulator 
COPY ./Simulator /simulator/Simulator

ENV PYTHONUNBUFFERED=1

# Start Xvfb virtual display then uvicorn
CMD Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99 && uvicorn Simulator.SimulatorServer:app --host 0.0.0.0 --port 8000