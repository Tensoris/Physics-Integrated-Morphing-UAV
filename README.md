# Morphing UAV Simulator: High-Fidelity Research Architecture

A modular Python-based framework designed to bridge the gap between analytical aerodynamics and high-cost CFD/FEM systems. This simulator targets research at the intersection of morphing wing structures, swarm intelligence, and physics-informed AI.

The system centers on a real-time **Digital Twin dashboard**, enabling interactive visualization of how wing morphing influences aerodynamic performance, structural response, and acoustic characteristics.


## Architecture Overview

The simulator is structured into five progressive research tiers:

### **Level 1 — High-Fidelity Aerodynamics**
- Numerical Vortex Lattice Method (VLM)
- Prandtl–Glauert compressibility corrections
- 3D wing mesh generation
- Real-time circulation and load distribution mapping

### **Level 2 — Structural FEM & Dynamics**
- Euler–Bernoulli beam theory (1D)
- Timoshenko beam theory (shear deformation effects)
- Aeroelastic deformation (bending and twist)
- Closed-loop PID controller for active gust rejection

### **Level 3 — Extreme Physics**
- Hypersonic thermal modeling (Sutton–Graves equation)
- Thermal expansion effects on structure
- Broadband aeroacoustic prediction (Ffowcs Williams–Hawkings)

### **Level 4 — Swarm Intelligence**
- Multi-agent UAV environment
- Flocking and coordination algorithms
- Procedural 3D terrain generation
- Nap-of-the-earth (NOE) navigation constraints

### **Level 5 — AI Surrogates & Digital Twin**
- Physics-Informed Neural Networks (PINNs)
- Real-time convergence monitoring
- Surrogate modeling for rapid evaluation
- Anomaly detection (flutter, structural instability)


## Installation

### Clone Repository
```bash
git clone https://github.com/yourusername/morphing-uav-simulator.git
cd morphing-uav-simulator
```
Create Virtual Environment
```
python -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\activate
```
Install Dependencies
```
pip install -r requirements.txt

```

Usage
```
Launch the Digital Twin dashboard:

python morphing-uav.py

Access the interface at:

http://127.0.0.1:8050/

Interactive Controls
	•	Wing span
	•	Sweep angle
	•	Twist distribution
	•	Morphing parameters

All updates propagate in real time across aerodynamic, structural, and acoustic models.

```

Data Export
```
The dashboard includes a Download Dataset feature.

Exported CSV includes:
	•	Spanwise lift distribution
	•	Circulation values
	•	Bending moments
	•	Aerodynamic coefficients

Designed for downstream analysis and integration into custom pipelines.

<br>
```

Citation
```
@software{morphing_uav_simulator_2026,
  author       = {Aaryan Chaulagain},
  title        = {Morphing UAV Simulator Suite: High-Fidelity Research Architecture},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19258496},
  url          = {https://doi.org/10.5281/zenodo.19258496}
}
```


License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

Permissions:
	•	Use, modify, and distribute
	•	Commercial and non-commercial applications

Condition:
	•	Attribution to the original author is required



Research Scope

This framework is intended for:
	•	Aeroelasticity research
	•	Morphing wing optimization
	•	Multi-agent UAV coordination
	•	Physics-informed machine learning
	•	Digital twin system development



Notes
	•	Designed with modular fallback support for heavy dependencies
	•	Scales from lightweight experimentation to advanced research workflows
	•	Compatible with JIT acceleration, RL environments, and surrogate modeling pipelines

