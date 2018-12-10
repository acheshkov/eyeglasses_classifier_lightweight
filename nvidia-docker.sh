#!/bin/bash
if [ -z "$NV_ARGS" ]; then
  # $NV_ARGS is not set, cannot mock nvidia-docker without it.
  echo "# \$NV_ARGS env variable required to mock nvidia-docker. Settable using e.g."
  echo "export NV_ARGS=\$(docker-machine ssh \$1 curl -s http://localhost:3476/docker/cli)"
elif [[ "$#" -gt "1" && "$1" -eq "run" ]]; then
  # Got run command with some args, add NV_ARGS and use it with docker.
  shift 1  # Drop first arg from $@.
  docker run $NV_ARGS $@
else
  # Every other time, just fall back to using docker.
  docker $@
fi
