from locust import HttpUser, task, between, constant, events
from datetime import datetime, timedelta, date
from random import randint
import random
import json
import uuid
import numpy as np
import logging
import sys
import time
import os
import string
import logging
from requests.adapters import HTTPAdapter 

VERBOSE_LOGGING = 0

def matrix_checker(matrix):
    sum = np.round(np.sum(matrix, axis=1),decimals=4)
    for i in range(10):
        if sum[i] < 0.99 or sum[i] > 1.01:
            return False
    return True


def sequence_generator(matrix, all_functions):

    if(not(matrix_checker(matrix))):
        raise Exception("Matrix is not correct")

    max_sequence_len = 20
    current_node = 0
    i = 0

    array = []
    array.append(all_functions[0])

    while(i < max_sequence_len):
        if(1 in matrix[current_node] and matrix[current_node].tolist().index(1) == current_node):
            break
        selection = random.choices(
            population=all_functions, weights=matrix[current_node])[0]
        array.append(selection)

        current_node = all_functions.index(selection)

        i += 1
    return array

@events.init_command_line_parser.add_listener
def _(parser):
    parser.add_argument("--matrix", type=str, env_var="matrix", default="", help="It's working") #Controllare

    
class Requests():

    def __init__(self, client):
        self.client = client

        dir_path = os.path.dirname(os.path.realpath(__file__))
        handler = logging.FileHandler(os.path.join(dir_path, "locustfile_debug.log"))   
        
        if VERBOSE_LOGGING==1:
            logger = logging.getLogger("Debugging logger")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            self.debugging_logger = logger
        else:
            self.debugging_logger = None

    def log_verbose(self, to_log):
        if self.debugging_logger!=None:
            self.debugging_logger.debug(json.dumps(to_log))

    def service(self, name):
        req_label = name
        start_time = time.time()
        with self.client.get(
                url = "/"+name,
                catch_response = True,
                name = req_label) as response:
            # LG: Devo aggiungere qui per ogni log il microservizio manualmente, togliere la response e magari formattare meglio.
            to_log = {'microservice':name, 'name': req_label, 'status_code': response.status_code, 'response_time': time.time() - start_time}
           
            self.log_verbose(to_log)
            
    def perform_task(self, name):
    	self.service(name)

class MuBench(HttpUser):
    weight = 1
    wait_time = constant(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client.mount('https://', HTTPAdapter(pool_maxsize=50))
        self.client.mount('http://', HTTPAdapter(pool_maxsize=50))

    @task
    def perfom_task(self):
        logging.debug("Running user 'muBench'...")
        all_functions = ["s0","s1","s2","s3","s4","s5","s6","s7","s8","s9"]
	
        #matrix = np.loadtxt(self.environment.parsed_options.matrix,dtype='d',delimiter=',')
        matrix = np.array([[0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05],
                  [0.55,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]])
                  
        task_sequence = sequence_generator(matrix, all_functions)
	 
        requests = Requests(self.client)
        for task in task_sequence:
        	requests.perform_task(task)
