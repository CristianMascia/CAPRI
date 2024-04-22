#!/bin/bash

#check if minikube is running and stop if it is running
if [ "$(docker container inspect -f '{{.State.Status}}' minikube)" == 'running' ]; then
  echo 'minikube is running'
  minikube stop
fi



#start minikube
echo "starting Minikube"
nmcli radio wifi off
minikube start --cpus=4 --memory 4096 --disk-size 32g --extra-config=kubelet.housekeeping-interval=10s
nmcli radio wifi on

minikube addons enable metrics-server 

#undeploy
cd configs/repo
skaffold delete
#kubectl delete -f configs/complete-demo.yaml
cd ../../

echo "Wait until internet connection is ready"
#wait until internet connection is ready
while ! wget -q --spider http://google.com; do
  : #nop
done

ulimit -n 100000 #extend limit of opened files

# Define variables
host=http://192.168.49.2:30123
n_nodes=11

#deploy
#kubectl create -f configs/complete-demo.yaml #> /dev/null
cd configs/repo
kubectl create namespace onlineboutique
skaffold run
#kubectl delete -f configs/complete-demo.yaml
cd ../../

echo "Waiting services and metrics are available"
while kubectl get pods    -n onlineboutique | awk '(NR>1)' | wc -l | grep -q -v $n_nodes  ||
      kubectl get pods -A -n onlineboutique | awk '(NR>1) {print $4}' | grep -v -q 'Running' ||
      kubectl get pods -A -n onlineboutique | awk '(NR>1) {print $3}' | grep -v -q '1/1\|2/2\|3/3' ||
      kubectl top pod     -n onlineboutique 2>/dev/null | awk '(NR>1)' | wc -l | grep -q -v $n_nodes ; do
   
  sleep 3
done

echo "Pods and Metrics are available"

#kubectl port-forward deployment/frontend 8080:8080 -n onlineboutique &
kubectl patch svc frontend -p '{"spec":{"externalIPs":["192.168.49.2"]}}' -n onlineboutique

#echo "WAIT"
#sleep 180
