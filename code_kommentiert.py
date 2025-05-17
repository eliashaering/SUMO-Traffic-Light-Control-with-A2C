# Bibliotheken für SUMO-Steuerung, Visualisierung, Datenstrukturen, maschinelles Lernen etc.
import traci  
import matplotlib.pyplot as plt  
import csv  
from collections import defaultdict 
import torch  
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F 
import numpy as np  

# Dictionaries zur Erfassung von Fahrzeugdaten
vehicle_wait_times = {}        # Speichert Wartezeit pro Fahrzeug
vehicle_drive_though = {}      # Speichert, wie lange Fahrzeuge durch Ampelbereiche gefahren sind

# Funktion zur Ermittlung der aktuellen Wartezeit je Fahrzeug an einer bestimmten Ampel
def get_waiting_times_at_traffic_light_step(tl_id, step_length=1):
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)  # Spuren, die zur Ampel gehören

    for vehicle_id in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(vehicle_id)  # Aktuelle Spur des Fahrzeugs
        speed = traci.vehicle.getSpeed(vehicle_id)     # Geschwindigkeit des Fahrzeugs
        
        if lane_id in controlled_lanes:
            if vehicle_id not in vehicle_drive_though:
                vehicle_drive_though[vehicle_id] = 0.0
            vehicle_drive_though[vehicle_id] += step_length  # Fahrtzeit auf Ampelspur wird addiert

        if lane_id in controlled_lanes and speed < 1:
            if vehicle_id not in vehicle_wait_times:
                vehicle_wait_times[vehicle_id] = 0.0
            vehicle_wait_times[vehicle_id] += step_length  # Wartezeit zählt, wenn Fahrzeug stillsteht

# Gibt die maximalen Wartezeiten je Spur an der Ampel zurück
def getWartezeit(tl_id):
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    max_waiting_times = {lane_id: 0.0 for lane_id in controlled_lanes}

    for vehicle_id in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        if lane_id in controlled_lanes:
            waiting_time = vehicle_wait_times.get(vehicle_id, 0.0)
            if waiting_time > max_waiting_times[lane_id]:
                max_waiting_times[lane_id] = waiting_time
    return list(max_waiting_times.values())

# Gibt die aufsummierte Gesamtwartezeit aller Fahrzeuge an der Ampel zurück
def getGesamtwartezeit(tl_id):
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    summe = 0
    for vehicle_id in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        if lane_id in controlled_lanes:
            waiting_time = vehicle_drive_though.get(vehicle_id, 0.0)
            summe += waiting_time
    return summe

# Neuronales Netz für A2C-Agenten mit Actor und Critic
class A2CNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(A2CNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 128)  # Gemeinsame Schicht für Feature-Extraktion
        self.actor = nn.Linear(128, action_dim)  # Ausgabe der Policy (Wahrscheinlichkeit je Aktion)
        self.critic = nn.Linear(128, 1)          # Ausgabe des Werts (für Wertschätzung)

    def forward(self, x):
        x = F.relu(self.fc(x))  # Aktivierung mit ReLU
        return self.actor(x), self.critic(x)

# A2C-Agent, der das Netz nutzt und das Lernen steuert
class A2CAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.model = A2CNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99  # Diskontierungsfaktor

    def reset_actor(self):
        # Initialisiert nur die Actor-Schicht neu, z.B. bei Training-Neustart
        for layer in [self.model.actor]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def select_action(self, state):
        state = torch.FloatTensor(state)
        logits, _ = self.model(state)
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()  # Wählt eine Aktion probabilistisch
        return action.item(), dist.log_prob(action)

    def train(self, trajectory):
        states, actions, log_probs, rewards, values = zip(*trajectory)

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)  # Rückwärts aufrollen (TD-Target)
        returns = torch.FloatTensor(returns)
        values = torch.cat(values)
        log_probs = torch.stack(log_probs)

        advantage = returns - values  # Berechnet den Vorteil (für Policy-Gradient)

        actor_loss = -torch.mean(log_probs * advantage.detach())  # Policy-Loss
        critic_loss = F.mse_loss(values, returns)                # Wert-Schätzung-Loss
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Bildung des State Space
def get_state():
    max_waiting_vehicles =  sum(traci.lane.getLastStepHaltingNumber(lane) for lane in list(set(controlled_lanes))) +1
    waiting_vehicles = [
        traci.lane.getLastStepHaltingNumber(lane) / max_waiting_vehicles for lane in list(set(controlled_lanes))
    ]   

    wartezeiten = getWartezeit(tl_id)
    max_wartezeit = max(wartezeiten) if wartezeiten else 1.0
    if max_wartezeit == 0:
        waiting_times = [0.0 for _ in wartezeiten]
    else:
        waiting_times = [w / max_wartezeit for w in wartezeiten]

    current_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
    phase_one_hot = [1 if c == "G" else 0 for c in current_phase]
    phase_duration = traci.trafficlight.getPhaseDuration(tl_id)

    
    # StateSpace Moeglichkeiten
    # Maximale Wartezeit
    #state_vector = waiting_times

    # Anzahl wartender Fahrzeuge
    #state_vector = waiting_vehicles

    # Aktuelle Ampelphase
    #state_vector = phase_one_hot

    # Phasendauer
    #state_vector = [phase_duration]

    # Kombination 
    state_vector =  waiting_vehicles +  waiting_times + phase_one_hot + [phase_duration]

    return state_vector

# Gruppiert Links (Verbindungen) je Spur
def group_links_by_lanes(tl_id):
    controlled_links = traci.trafficlight.getControlledLinks(tl_id)
    grouped = defaultdict(list, { 
        ('814691984_1', '31342064_1'): [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 20], 
        ('25796398_1','137463477_1'): [7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26], 
        ('814691984_2'): [5], 
        ('31342064_2'): [19], 
        ('137463477_2'): [12, 13], 
        ('25796398_2'): [26], 
    })
    return grouped

# Setzt den Ampelzustand anhand aktiver Phasen (Grün/Rotor)
def set_trafficlight_state(tl_id, green_indices, green_char="G", red_char="r"):
    current_state = list(traci.trafficlight.getRedYellowGreenState(tl_id))
    for i in range(len(current_state)):
        current_state[i] = green_char if i in green_indices else red_char
    new_state = "".join(current_state)
    traci.trafficlight.setRedYellowGreenState(tl_id, new_state)

# SUMO-Simulation starten
sumo_cmd = ["sumo", "-c", "Hier den Pfad zur Sumo-Simulation eintragen","--no-warnings", "--verbose", "false"]
traci.start(sumo_cmd)

# Ampel-ID setzen
tl_id = "cluster_25579770_2633530003_2633530004_2633530005"
grouped_lanes = group_links_by_lanes(tl_id)

# Index für die erste Spur
lane_index = 0
controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
num_lanes = len(controlled_lanes)
lane_id = controlled_lanes[lane_index]

# Initialisierung des A2C-Agenten
# Die state_dim muss je nach StateSpace-Variante unterschiedliche parametriert werden

# Maximale Wartezeit
#state_dim = len(list(set(controlled_lanes)))

# Anzahl wartender Fahrzeuge
#state_dim = len(list(set(controlled_lanes)))

# Aktuelle Ampelphase
#state_dim = len(traci.trafficlight.getRedYellowGreenState(tl_id))

# Phasendauer
#state_dim = 1

# Kombination 
state_dim = len(list(set(controlled_lanes))) * 2 + len(traci.trafficlight.getRedYellowGreenState(tl_id)) + 1

# Weitere Initialisierung
action_dim = len(grouped_lanes)
agent = A2CAgent(state_dim, action_dim)
agent.reset_actor()
trajectory = []

# CSV-Datei vorbereiten zur Datenspeicherung
csv_filename = "traffic_data.csv"
with open(csv_filename, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Zeit (s)", "Wartende Autos", "Max. Wartezeit (s)", "Durch. Wartezeit(s)"])

# Initialisierung von Variablen für Diagramme und Statistiken
time_steps = []
waiting_vehicles_list = []
max_waiting_time_list = []
reward_list = []
longest_waiting_lane_over_time = []
max_list = []
max_list_1 = []
avg_wait_times = []
num_veh = []

# Weitere Steuerparameter
is_green = True  
simulation_time = 4001  
step = 1
prev_waiting_time = 0
vehicles_passed = 0
last_vehicle_ids = set()

# Haupt-Simulationsschleife
while step < simulation_time:
    traci.simulationStep()  # Nächster Schritt

    if step % 5 == 0:
        state = get_state()
        action_idx, log_prob = agent.select_action(state)
        group_keys = list(grouped_lanes.keys())
        selected_group = group_keys[action_idx]
        green_indices = grouped_lanes[selected_group]
        set_trafficlight_state(tl_id, green_indices)
        value = agent.model(torch.FloatTensor(state))[1]

        # Reward-Berechnung (hier: Anzahl wartender Fahrzeuge minimieren)
        waiting_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
        reward = -waiting_vehicles

        trajectory.append((state, action_idx, log_prob, reward, value))
        reward_list.append(reward)

    if step % 100 == 0 and len(trajectory) > 0:
        agent.train(trajectory)
        trajectory.clear()

    if step % 4 == 0:
        get_waiting_times_at_traffic_light_step(tl_id)

    if step % 10 == 0:
        wartezeiten = getWartezeit(tl_id)
        max_waiting_time = max(wartezeiten)
        waiting_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes)
        average_waiting = getGesamtwartezeit(tl_id) / len(wartezeiten) if wartezeiten else 0

        time_steps.append(step)
        waiting_vehicles_list.append(waiting_vehicles)
        max_waiting_time_list.append(max_waiting_time)
        max_list_1.append(max_waiting_time)

        with open(csv_filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([step, waiting_vehicles, max_waiting_time, average_waiting])

        lane_wait_times = {lane: traci.lane.getWaitingTime(lane) for lane in controlled_lanes}
        longest_lane = max(lane_wait_times, key=lane_wait_times.get)
        longest_waiting_lane_over_time.append(longest_lane)

    if step % 4000 == 0:
        agent.reset_actor()
        if vehicle_drive_though:
            avg_waiting_time = sum(vehicle_drive_though.values()) / len(vehicle_drive_though.values())
        else:
            avg_waiting_time = 0
        avg_wait_times.append(avg_waiting_time)
        num_veh.append(len(vehicle_drive_though.values()))
        max_list.append(max(vehicle_wait_times.values()))
        max_list_1 = []
        vehicle_wait_times = {}
        vehicle_drive_though = {}

    step += 1

traci.close()

# Speichern der Auswertungsdaten
with open('output_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Durchschnittliche Wartezeit", "Anzahl der Fahrzeuge"])
    writer.writerows(zip(avg_wait_times, num_veh))

with open('output_data1.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Maximale Wartezeit", "Anzahl der Fahrzeuge"])
    writer.writerows(zip(max_list, num_veh))

