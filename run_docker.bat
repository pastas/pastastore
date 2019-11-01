@echo off
REM I ran the following command first time to ensure data was persistent
docker volume create --name=mongodata
docker run --mongodb -v mongodata:/data/db -d -p 27017:27017 mongo

REM Other snippets:
REM ===============
REM Create backup to somewhere...
REM docker run --rm --volumes-from dbstore -v /backup ubuntu tar cvf /backup/backup.tar data/db

REM Copy data from mondodb docker volume to local disk
REM docker cp 7d549a84df43:/data/db /data/backup/