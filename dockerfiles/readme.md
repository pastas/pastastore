# Running MongoDB from docker
This readme explains how to get the MongoDB docker container up and running.

## Using `docker-compose`
Create and start the MongoDB docker container using `docker-compose`.
The `docker-compose.yml` file describes the container.
1. Install Docker (i.e.
   [Docker Desktop](https://www.docker.com/products/docker-desktop))
2. Open a terminal and navigate to `/dockerfiles`.
3. Run `docker-compose up -d` to start the docker container running MongoDB.
   The `-d` flag runs the container in the background. This command uses the
   `docker-compose.yml` file by default.
4. View your runing containers with `docker ps -a`.


To stop the container run `docker-compose stop` in a terminal.

Once the docker container exists, starting it using the `docker-compose up`
command might fail. Use `docker start mongodb` from the terminal instead.
Check if it is running using `docker ps -a`.

<hr>

## Manual set-up
The following steps describe the manual set-up of the docker container for
MongoDB. It is recommended to use the `docker-compose` method above, but this
is left here as a reference.
1. Install Docker (i.e.
   [Docker Desktop](https://www.docker.com/products/docker-desktop)).
2. Open a terminal and pull the mongodb image by typing `docker pull mongo`.
3. To ensure a persistent volume (i.e. your data is kept even if you shutdown
   your docker container running mongodb) run the following command before
   starting the mongo database, e.g. `docker volume create --name=mongodata`
4. Start the docker container on localhost by running e.g.
   `docker run --mongodb -v mongodata:/data/db -d -p 27017:27017 mongo`
5. View your running containers with `docker ps -a`

Stop the container with the command: `docker stop mongodb`.

(Re)start the container (after it has been created) by typing
`docker start mongodb` in a terminal.
