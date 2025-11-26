# Adaptive Verkehrssteuerung mit A2C und SUMO Test

Dieses Projekt implementiert eine adaptive Ampelsteuerung basierend auf Reinforcement Learning (A2C - Advantage Actor Critic) in Verbindung mit der mikroskopischen Verkehrssimulation SUMO.

Ziel ist es, die Wartezeiten und Staus an einer Kreuzung durch lernende Algorithmen zu minimieren.

## Voraussetzungen

- [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) muss lokal installiert sein.
- Python 3.x
- Benötigte Python-Bibliotheken:
  - `traci`
  - `matplotlib`
  - `torch`
  - `numpy`

## Simulationsdaten
Für die Simulation wurden die realen Verkehrsdaten der Stadt Ingolstadt verwendet. Diese sind verfügbar unter:
https://github.com/TUM-VT/sumo_ingolstadt

Die angepasste Version einzelner Konfigurationsdateien ist im Zip-Ordner "Simualationsdaten" zu finden 

Bitte laden Sie das Repository herunter und passen Sie den Pfad zur Konfigurationsdatei (.sumocfg) im Skript entsprechend an.

## Anpassung des State Spaces
Im Code kann der State Space flexibel gewählt werden, um unterschiedliche Aspekte der Verkehrssituation abzubilden. Die möglichen State Spaces finden Sie im Abschnitt, indem die Variable state_dim gesetzt wird. Dort sind verschiedene Varianten auskommentiert:

- Maximale Wartezeit pro Spur
- Anzahl wartender Fahrzeuge pro Spur
- Aktuelle Ampelphase (one-hot codiert)
- Dauer der aktuellen Ampelphase
- Kombination aus den oben genannten Zuständen

Durch das Aus- oder Einkommentieren der jeweiligen Zeilen können Sie den Zustand an Ihre Anforderungen anpassen.
Durch Aus- und Einkommentieren

## Verwendung
Starten Sie die SUMO-Simulation über das Skript, das die Steuerung übernimmt. Der A2C-Agent lernt während der Simulation, die Ampelphasen so zu steuern, dass Wartezeiten minimiert werden.

Die Ergebnisse (z.B. Wartezeiten und Fahrzeugzahlen) werden in CSV-Dateien gespeichert und können zur weiteren Analyse verwendet werden.

## Hinweise
- Achten Sie darauf, dass der Pfad zur SUMO-Konfigurationsdatei im Skript korrekt gesetzt ist.
- Die Bibliothek traci ist eine Schnittstelle, die SUMO-Steuerbefehle in Python ermöglicht.
- Die Trainingsparameter des A2C-Agenten können ebenfalls im Skript angepasst werden.
- Um eine klasssiche Ampelsteuerung zu fordern müssen im Code die Zeilen 2040 bis 255 auskommentiert werden


