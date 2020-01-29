@echo off
REM Ensure you have the mongo docker image, if not type `docker pull mongo` first

REM The following command ensures data is persistent (doesn't disappear after
REM stopping mongodb)
docker volume create --name=mongodata
REM Run mongodb
docker run --mongodb -v mongodata:/data/db -d -p 27017:27017 mongo

REM After setting up the mongodb container with the previous commands,
REM Restarting the container is as simple as:
REM `docker start mongodb`

REM Other snippets:
REM ===============
REM Create backup to somewhere...
REM docker run --rm --volumes-from dbstore -v /backup ubuntu tar cvf /backup/backup.tar data/db

REM Copy data from mondodb docker volume to local disk
REM docker cp 7d549a84df43:/data/db /data/backup/