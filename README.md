# MPC Trajectory Tracking Framework


This repository contains the implementation of the Linear Model Predictive Control framework described in the conference paper: "Data-driven Model Predictive Control for Trajectory Tracking of Linear Systems With Multilayer Perceptron".

This project presents one of the simulations mentioned in paper.

## Project Structure

```text
mpc-project/
│
├── data/                  # generated databases (.db) for data-driven modeling
├── models/                # Trained neural network models (.h5) and scalers (.pkl)
│
├── src/                   # Core library
│   ├── __init__.py
│   ├── system.py          # System dynamics definitions (2D, 3D Kinematic, 3D Linear)
│   ├── control.py         # Generic MPC solver (CasADi-based)
│   ├── trajectory.py      # Reference trajectory generators
│   └── visualization.py   # Plotting and analysis tools
│
├── generate_data.py       # Script to simulate systems and generate training data
├── train.py               # Script to train neural network dynamics models
├── main.py                # Main entry point for running control simulations
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Install dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

## Usage

### Generate Training Data

```bash
python generate_data.py
```

This creates a SQLite database in `data/system_dynamics_3_2.db` containing state-action pairs.

### Train the Model

```bash
python train.py
```

Models will be saved to the `models/` directory.

### Run the Control Simulation

To run the standard trajectory tracking simulation with the default configuration:

```bash
python main.py
```

## Results

The simulation produces tracking performance plots saved or displayed at the end of the run, showing:

1. XY-Plane Trajectory: Reference vs. Actual path.
2. Control Inputs: Velocity and Angular Velocity over time.
3. Tracking Error: Euclidean distance error metric.
