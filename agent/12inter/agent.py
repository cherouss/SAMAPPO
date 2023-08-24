from itertools import cycle
import os
import sys
import random
import numpy as np
from collections import  defaultdict
import torch
#import gym
from sumolib import checkBinary 
import traci  

def normalize(arr):
    normalized = (arr-np.amin(arr))/(np.amax(arr) - np.amin(arr))
    return normalized*2 - 1

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def get_adj_matrix(doc):
    result = np.eye(12,12)
    name_to_index = {}
    i = 0
    for k in doc.keys():
        name_to_index[k] = i
        i+=1

    for k,v in doc.items():
        for i in v:
            result[name_to_index[k],name_to_index[i]] =1
    return result


class Agent():
    def __init__(self,act_dim = 2 ,state_dim = 3,data = 0):
        #self.observation_space = gym.spaces.Discrete(state_dim)
        #self.action_space = gym.spaces.Discrete(act_dim)
        self.phases = [cycle([0,2]) for _ in range(12)]
        self.act_dim = act_dim
        self.his = [[0,0]]
        self.state_dim = state_dim
        self.terminate = False
        self.waiting_time = 0
        
        self.sim_step = 0
        self.current_phase_duration = 0
        self.junction_connections = {
                'J15':('gneJ10','gneJ14'),
                'J4' : ('gneJ14','gneJ20'), 
                'gneJ10' : ('gneJ13','gneJ6','J15' ),
                'gneJ11' : ('gneJ13','gneJ8','gneJ6','gneJ22'),
                'gneJ13' : ('gneJ10','gneJ11','gneJ14','gneJ20'),
                'gneJ14' : ('gneJ13','J15','J4' ),
                'gneJ20' : ('gneJ13','gneJ22','J4'),
                'gneJ21' : ('gneJ8','gneJ22'),
                'gneJ22' : ('gneJ21', 'gneJ20','gneJ11'),
                'gneJ5' : ('gneJ6','gneJ8'),
                'gneJ6' : ('gneJ5','gneJ10','gneJ11'),
                'gneJ8' : ('gneJ5','gneJ11','gneJ21'),}
        self.tls_detectors = {'J15': ['e2det_-E21_0', 'e2det_-E21_1', 'e2det_-E0_0', 'e2det_E17_0', 'e2det_E17_1', 'e2det_E3_0'], 
 'J4': ['e2det_E2_0', 'e2det_-E9_0', 'e2det_-E9_1', 'e2det_-E13_0', 'e2det_-E5_0', 'e2det_-E5_1'], 
 'gneJ10': ['e2det_E4_0', 'e2det_-E17_0', 'e2det_-E17_1', 'e2det_gneE20_0', 'e2det_gneE12_0', 'e2det_gneE12_1'],
 'gneJ11': ['e2det_gneE14_0', 'e2det_gneE21_0', 'e2det_gneE21_1', 'e2det_gneE39_0', 'e2det_gneE23_0', 'e2det_gneE23_1'], 
 'gneJ13': ['e2det_gneE18_0', 'e2det_gneE25_0', 'e2det_gneE25_1', 'e2det_gneE35_0', 'e2det_gneE24_0', 'e2det_gneE24_1'],
 'gneJ14': ['e2det_E0_0', 'e2det_-E1_0', 'e2det_-E1_1', 'e2det_-E2_0', 'e2det_gneE26_0', 'e2det_gneE26_1'],
 'gneJ20': ['e2det_E5_0', 'e2det_E5_1', 'e2det_-E12_0', 'e2det_-E6_0', 'e2det_-E6_1', 'e2det_gneE34_0'], 
 'gneJ21': ['e2det_E7_0', 'e2det_E7_1', 'e2det_-E10_0', 'e2det_-E8_0', 'e2det_-E8_1', 'e2det_gneE36_0'], 
 'gneJ22': ['e2det_gneE38_0', 'e2det_E6_0', 'e2det_E6_1', 'e2det_-E11_0', 'e2det_-E7_0', 'e2det_-E7_1'], 
 'gneJ5': ['e2det_gneE11_0', 'e2det_gneE4_0', 'e2det_gneE4_1', 'e2det_gneE9_0', 'e2det_gneE5_0', 'e2det_gneE5_1'],
 'gneJ6': ['e2det_gneE17_0', 'e2det_gneE13_0', 'e2det_gneE13_1', 'e2det_gneE15_0', 'e2det_gneE3_0', 'e2det_gneE3_1'],
 'gneJ8': ['e2det_gneE7_0', 'e2det_gneE22_0', 'e2det_gneE22_1', 'e2det_gneE37_0', 'e2det_gneE29_0', 'e2det_gneE29_1']}
        
        self.tls_lanes = {'gneJ10': ('gneE27_0','gneE27_1', 'gneE20_0','gneE12_0','gneE12_1','gneE31_0', ),
                'gneJ11': ('gneE14_0','gneE21_0','gneE21_1','gneE39_0','gneE23_0','gneE23_1'),
                'gneJ13': ('gneE18_0','gneE25_0','gneE25_1','gneE35_0','gneE24_0','gneE24_1'),
                'gneJ5': ('gneE11_0','gneE4_0','gneE4_1',  'gneE9_0','gneE5_0','gneE5_1'),
                'gneJ6': ('gneE17_0','gneE13_0','gneE13_1','gneE15_0','gneE3_0','gneE3_1'),
                'gneJ8': ('gneE7_0','gneE22_0','gneE22_1','gneE37_0','gneE29_0','gneE29_1')}
        #import scipy.sparse as sps
        #from torch_geometric.utils import from_scipy_sparse_matrix
        #self.edge_index = from_scipy_sparse_matrix(sps.coo_matrix(get_adj_matrix(self.junction_connections)))
        self.edge_index = get_adj_matrix(self.junction_connections).tolist()
        self.link = f"./{data}/osm.sumocfg"
        #print(self.edge_index)



        #self.intersection_id 
        
        
        self.traci = traci
        self.intersection_id  =  None
        self.CO2_emission = 0
        #generate_routefile()
        self.state = (0,0,0,0,0,0,0,0,0)
        

    def reset(self):
        label = 0
        done = False
        self.waiting_time = 0
        
        while not done:
            try:
                self.traci.start(['sumo', "-c", self.link, ],label=str(label))
                done = True
            except:
                label+=1
        self.terminate  = False
        self.traci.simulationStep(30)
        self.sim_step +=30

        #self.intersection_id= self.traci.trafficlight.getIDList()
        self.CO2_emission = 0
        #self.phases = cycle([1,3,5,7])
        self.phases = [cycle([0,2]) for _ in range(12)]
    def get_reward(self):
        waiting_ = 0
        halt = 0
        rewards = []
        #t = self.traci.lane.getIDList()
        t = self.traci.lanearea.getIDList()
            #print(t)
            #print(f'emission : {carbon_omission}')
        for i in t:
                #occup+=traci.lane.getLastStepOccupancy(i)
            halt += self.traci.lanearea.getJamLengthVehicle(i)
        vehicles = self.traci.vehicle.getIDList()

        for vhc in vehicles:
            waiting_ += self.traci.vehicle.getAccumulatedWaitingTime(vhc)/len(vehicles)
               
        #waiting_ = waiting_/count
        #fuel = fuel/count
        #res = occup + halt
        if halt and waiting_:
                return 5000 - (halt*waiting_)
        return 0
        #return np.array(rewards).mean()
          
    def _get_system_info(self):
        vehicles = self.traci.vehicle.getIDList()
        speeds = [self.traci.vehicle.getSpeed(vehicle) for vehicle in vehicles]
        waiting_times = [self.traci.vehicle.getWaitingTime(vehicle) for vehicle in vehicles]
        return {
            # In SUMO, a vehicle is considered halting if its speed is below 0.1 m/s
            "system_total_stopped": sum(int(speed < 0.1) for speed in speeds),
            "system_total_waiting_time": sum(waiting_times),
            "system_mean_waiting_time": 0.0 if len(vehicles) == 0 else np.mean(waiting_times),
            "system_mean_speed": 0.0 if len(vehicles) == 0 else np.mean(speeds),
        }

    
            

    
    
    def step(self,act):
        self.action(act,self.junction_connections.keys())
        self.sim_step +=4
        self.traci.simulationStep(self.sim_step)
        state,reward = self.get_state(),self.get_reward()
        #a = len(self.traci.vehicle.getIDList())
        for v in self.traci.vehicle.getIDList():
            self.waiting_time+=self.traci.vehicle.getAccumulatedWaitingTime(v)
            co2_emission = self.traci.vehicle.getCO2Emission(v)
            self.CO2_emission += co2_emission
        if self.sim_step >= 3600 or self.traci.simulation.getMinExpectedNumber() <= 0:
            print(f'\n\nlast sim insede step {self.sim_step}')
            self.terminate = True
            self.sim_step = 0
        if self.terminate:
            self.his.append([self.waiting_time,self.CO2_emission])
            self.traci.close() 
            sys.stdout.flush()
        info = _get_system_info()
        return state , reward , self.terminate , info
    
    def get_state(self):
        lst = []
        res = []
        #det = traci.lanearea.getIDList()
        junction_connections = self.tls_detectors
        j = 0
        for jun,det in junction_connections.items():
            phase = self.traci.trafficlight.getPhase(jun)
            current_phase_duration = self.traci.trafficlight.getPhaseDuration(jun) - \
                                (self.traci.trafficlight.getNextSwitch(jun) -
                                    self.traci.simulation.getTime())
            lst = [int(phase),current_phase_duration]
            sm = 0
            for i in det:
                #print(i)
                sm += self.traci.lanearea.getLastStepHaltingNumber(i)
            lst.append(sm)
            lst+= self.edge_index[j]
            res.append(lst)
            j+=1
        return torch.tensor(res).float(),
    
    def action(self,act,tls):
        i = 0
        for intersection_id in tls:
            #phase = self.traci.trafficlight.getPhase(intersection_id)
            phase = self.traci.trafficlight.getRedYellowGreenState(intersection_id)
            if 'y' in phase or 'Y' in phase:#let yellows stays
                    continue
            if act[i]:
            
                self.traci.trafficlight.setPhaseDuration(intersection_id,4)
            else :
                
                self.current_phase_duration = self.traci.trafficlight.getPhaseDuration(intersection_id) - \
                                (self.traci.trafficlight.getNextSwitch(intersection_id) -
                                    self.traci.simulation.getTime())
                if self.current_phase_duration <= 4:
                    continue
                else:
                    self.traci.trafficlight.setPhase(intersection_id,next(self.phases[i]))
            i+=1
            
        
        
    def get_observation(self):
        state = self.get_state()
        reward = self.get_reward()
        #reward = np.clip(-100,100,self.get_reward())
        return state , reward , self.terminate
            
