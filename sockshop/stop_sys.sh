#!/bin/bash

#Undeploy pods
kubectl delete -f configs/complete-demo.yaml
minikube stop


