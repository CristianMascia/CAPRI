#!/bin/bash

#Undeploy pods
#kubectl delete -f configs/complete-demo.yaml
cd configs/repo
skaffold delete
kubectl delete namespace onlineboutique
#kubectl delete -f configs/complete-demo.yaml
cd ../../
minikube stop


