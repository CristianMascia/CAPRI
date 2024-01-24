#!/bin/bash

#IMP: si considera che mubench è stato gia installato


#check if mubench is running and stop if it is running
if [ "$(docker container inspect -f '{{.State.Status}}' mubench)" == 'running' ]; then
  echo 'muBench is running'
  docker stop mubench
fi

#check if minikube is running and stop if it is running
if [ "$(docker container inspect -f '{{.State.Status}}' minikube)" == 'running' ]; then
  echo 'minikube is running'
  minikube stop
fi

minikube addons enable metrics-server 

#start minikube and muBench
echo "starting Minikube"
nmcli radio wifi off
minikube start --memory 8192 --cpus 4  --extra-config=kubelet.housekeeping-interval=10s
nmcli radio wifi on
echo "starting muBench"
docker start mubench

#check if there are pods
a=0
b=$(minikube kubectl get pods | wc -l)
if [[ $b -gt $a ]]; then
  #Undeploy existing pods
  docker exec mubench /bin/bash -c 'yes y | python3 Deployers/K8sDeployer/RunK8sDeployer.py -c Configs/K8sParameters.json'
fi

echo "Wait until internet connection is ready"

while ! wget -q --spider http://google.com; do
  : #nop
done

ulimit -n 100000 #extend limit of opened files

# Define variables
host=http://192.168.49.2:31113
n_nodes=$((10 + 1)) #10 servizi, 1 gw-ginx

#backup e load of configuration file
echo "BACKUP PARAMETRI"
docker cp mubench:/root/muBench/Configs/K8sParameters.json configs/K8sParameters_backup.json
docker cp configs/K8sParameters.json mubench:/root/muBench/Configs/K8sParameters.json

#upload workmodel
docker cp configs/workmodel.json mubench:/root/muBench/Configs/workmodel.json
#deploy
docker exec mubench python3 Deployers/K8sDeployer/RunK8sDeployer.py -c Configs/K8sParameters.json #> /dev/null

echo "Waiting services and metrics are available"
while minikube kubectl get pods | awk '(NR>1)' | wc -l | grep -q -v $n_nodes ||
  minikube kubectl -- get pods -A | awk '(NR>1) {print $4}' | grep -v -q 'Running' ||
  minikube kubectl -- get pods -A | awk '(NR>1) {print $3}' | grep -v -q '1/1' ||
  minikube kubectl top pod 2>/dev/null | awk '(NR>1)' | wc -l | grep -q -v $n_nodes; do
  sleep 3
done

echo "Pods and Metrics are available"
