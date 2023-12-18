#!/bin/bash

path_repo=../train-ticket

#Undeploy pods
(cd $path_repo ; make reset-deploy Namespace=trainticket)
minikube stop


