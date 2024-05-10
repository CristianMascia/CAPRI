#!/bin/bash

#Undeploy pods
kubectl delete pods,deployments,services -l app=teastore -n teastore
kubectl delete namespace teastore
minikube stop


