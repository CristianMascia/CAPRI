#!/bin/bash

#Undeploy pods
docker exec mubench /bin/bash -c 'yes y | python3 Deployers/K8sDeployer/RunK8sDeployer.py -c Configs/K8sParameters.json' > /dev/null

#Restore Backup
docker cp configs/K8sParameters_backup.json mubench:/root/muBench/Configs/K8sParameters.json
docker exec mubench rm /root/muBench/Configs/workmodel.json
rm configs/K8sParameters_backup.json

