#!/bin/bash

path_repo=../train-ticket

#check if minikube is running and stop if it is running
if [ "$(docker container inspect -f '{{.State.Status}}' minikube)" == 'running' ]; then
  echo 'minikube is running'
  minikube stop
fi

minikube delete
minikube config set memory 26000
minikube config set cpus 14


#start minikube
echo "starting Minikube"
nmcli radio wifi off
minikube start --extra-config=kubelet.housekeeping-interval=10s
nmcli radio wifi on



echo "Wait until internet connection is ready"
#wait until internet connection is ready
while ! wget -q --spider http://google.com; do
  : #nop
done

minikube addons enable metrics-server 

ulimit -n 100000 #extend limit of opened files

# Define variables
host=http://192.168.49.2:32677
n_pods=56

#deploy
kubectl create namespace trainticket
(cd $path_repo ; make deploy Namespace=trainticket)

echo "Waiting services and metrics are available"
while kubectl get pods -n trainticket | awk '(NR>1)' | wc -l | grep -q -v $n_pods  ||
   kubectl get pods -A -n trainticket | awk '(NR>1) {print $4}' | grep -v -q 'Running' ||
   kubectl get pods -A -n trainticket | awk '(NR>1) {print $3}' | grep -v -q '1/1\|2/2\|3/3' ||
   kubectl top pod -n trainticket 2>/dev/null | awk '(NR>1)' | wc -l | grep -q -v $n_pods ; do
   
  sleep 3
done

echo "Pods and Metrics are available"

