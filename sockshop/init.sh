#!/bin/bash


#check if minikube is running and stop if it is running
if [ "$(docker container inspect -f '{{.State.Status}}' minikube)" == 'running' ]; then
  echo 'minikube is running'
  minikube stop
fi



#start minikube
echo "starting Minikube"
nmcli radio wifi off
minikube start --memory 8192 --cpus 4 --extra-config=kubelet.housekeeping-interval=10s
nmcli radio wifi on

minikube addons enable metrics-server 

#undeploy
kubectl delete -f configs/complete-demo.yaml


echo "Wait until internet connection is ready"
#wait until internet connection is ready
while ! wget -q --spider http://google.com; do
  : #nop
done

ulimit -n 100000 #extend limit of opened files

# Define variables
host=http://192.168.49.2:30001
n_nodes=14

#deploy
kubectl create -f configs/complete-demo.yaml #> /dev/null

echo "Waiting services and metrics are available"
while kubectl get pods    -n sockshop | awk '(NR>1)' | wc -l | grep -q -v $n_nodes  ||
      kubectl get pods -A -n sockshop | awk '(NR>1) {print $4}' | grep -v -q 'Running' ||
      kubectl get pods -A -n sockshop | awk '(NR>1) {print $3}' | grep -v -q '1/1\|2/2\|3/3' ||
      kubectl top pod     -n sockshop 2>/dev/null | awk '(NR>1)' | wc -l | grep -q -v $n_nodes ; do
   
  sleep 3
done

echo "Pods and Metrics are available"

echo "WAIT"
sleep 180
