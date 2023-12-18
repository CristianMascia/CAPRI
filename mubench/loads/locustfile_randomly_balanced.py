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
        matrix = np.array([[0.0723,0.0675,0.1088,0.1046,0.1285,0.1592,0.1276,0.0395,0.0347,0.1573],
                  [0.0416,0.1311,0.1271,0.0961,0.1072,0.1027,0.1280,0.1121,0.1236,0.0305],
                  [0.1486,0.0688,0.1184,0.1592,0.0552,0.0629,0.0951,0.1422,0.0370,0.1127],
                  [0.0502,0.0924,0.1452,0.0911,0.0575,0.1342,0.1386,0.0895,0.0858,0.1154],
                  [0.0606,0.1399,0.0841,0.1628,0.0539,0.1164,0.0651,0.1028,0.1494,0.0650],
                  [0.1578,0.1212,0.0384,0.0595,0.0761,0.1532,0.1349,0.0911,0.0489,0.1188],
                  [0.0818,0.1318,0.1242,0.0392,0.1104,0.0738,0.1355,0.0692,0.1371,0.0970],
                  [0.0644,0.1221,0.1362,0.0461,0.1356,0.1252,0.1039,0.0768,0.0857,0.1039],
                  [0.1155,0.0668,0.0894,0.0443,0.1436,0.0638,0.1609,0.1448,0.1097,0.0612],
                  [0.0429,0.1309,0.1231,0.1092,0.1611,0.0843,0.0716,0.1212,0.0443,0.1113]])
                  
        task_sequence = sequence_generator(matrix, all_functions)
	 
        requests = Requests(self.client)
        for task in task_sequence:
        	requests.perform_task(task)
