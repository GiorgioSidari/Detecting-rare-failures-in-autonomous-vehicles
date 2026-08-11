FROM python:3.8.20-bookworm

WORKDIR /simulator

RUN apt-get update && apt-get install -y \
    libgl1 xvfb \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
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
# Force Mesa software rendering (llvmpipe): there is no GPU under Xvfb/emulation,
# so Unity must use the software GL path to obtain a context.
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=llvmpipe

# Start Xvfb virtual display then uvicorn.
# - rm -f the stale X lock so a restart (restart: unless-stopped) never gets stuck
#   on "Server is already active for display 99".
# - GLX/render extensions + sleep give the Unity simulator a working software-GL
#   display to attach to (the app forces graphical mode, so a display is required).
# Xvfb virtual display: low resolution (320x240) because Unity renders via software GL
# (llvmpipe) and the render cost is proportional to the pixels - fewer pixels, faster frames,
# higher control rate. Tunable via the XVFB_RESOLUTION env var.
ENV XVFB_RESOLUTION=320x240x24
CMD rm -f /tmp/.X99-lock; \
    Xvfb :99 -screen 0 ${XVFB_RESOLUTION} -ac +extension GLX +extension RANDR +render -noreset & \
    sleep 3 && export DISPLAY=:99 && \
    uvicorn Simulator.SimulatorServer:app --host 0.0.0.0 --port 8000