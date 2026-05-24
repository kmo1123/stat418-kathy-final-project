After installing podman and starting it, run:
podman build --platform linux/amd64 -f dockerfile -t ucla-study-api .


Then:
podman tag ucla-study-api docker_hub_username/ucla-study-api