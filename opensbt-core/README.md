# Useful stuff to know

## Setup

In order to run the package correctly it is needed to add to the 'opensbt-core' folder the Udacity Simulator executable and the model driver file.
These files where made available to us by [Davide Yi Xian Hu](https://dragonbanana.github.io/) on [GoogleDrive](https://drive.google.com/drive/folders/11sQXycm6hsv7fq6EUtZ8kEttJ4gPc-_t?usp=sharing)

- Add the Udacity Simulator folder containing the ubuntu executables in 'opensbt-core/Simulator/SimulatorExec/ubuntu_bianries/'

- Add the driver model file in 'opensbt-core/Simulator/SelfDrivingModels/'

## Simulator service

The simulator is now available as a service. To run it you just need to do:

```bash
docker compose up
```

This will launch the simulation server exposing the necessary APIs @ <http://0.0.0.0:8000>

To see the OpenApi documentation you can go to [/docs](http://127.0.0.1:8000/docs)

You can enqueue a simulation by posting the simulation to `/simulate`. This will return a jobId. You can then check the status/result of the simulation with a get request to `/simulate/{job_id}`
