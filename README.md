Morphing UAV Simulator: High-Fidelity Research Architecture
_Note: Must be deployed in local server_

Welcome! This simulator is a modular Python-based framework built to bridge the gap between simple analytical models and heavy-duty CFD/FEM suites. It is designed specifically for researchers exploring the intersection of morphing wing structures, swarm intelligence, and physics-informed AI.
The core of this project is a real-time, human-in-the-loop Digital Twin dashboard that lets you visualize how changing a wing's shape affects its performance, stability, and even its acoustic signature.
Research Tiers
The architecture is organized into five progressive levels, moving from basic physics to complex autonomous systems:
Level 1: High-Fidelity Aerodynamics Goes beyond basic approximations using a true Numerical Vortex Lattice Method (VLM). It includes Prandtl-Glauert compressibility corrections and generates a 3D wing mesh mapped with real-time circulation loads.
Level 2: Structural FEM & Dynamics Simulates how the wing actually bends and twists using 1D Euler-Bernoulli and Timoshenko beam theory. It also features a closed-loop PID controller designed to test active gust rejection through morphing.
Level 3: Extreme Physics For high-speed or high-altitude studies, this tier calculates hypersonic thermal heating (Sutton-Graves), structural expansion, and broadband noise (Ffowcs Williams-Hawkings).
Level 4: Swarm Intelligence A multi-agent environment where UAVs use flocking algorithms to navigate procedural 3D terrain while maintaining nap-of-the-earth (NOE) clearance.
Level 5: AI Surrogates & Digital Twin This is the "brain" of the system, providing live convergence monitoring for Physics-Informed Neural Networks (PINNs) and real-time anomaly detection for structural risks like flutter.
Setting Up
Clone the project:
git clone [https://github.com/yourusername/morphing-uav-simulator.git](https://github.com/yourusername/morphing-uav-simulator.git)
cd morphing-uav-simulator


Prepare your environment:
I recommend using a virtual environment to keep things clean:
python -m venv venv
source venv/bin/activate  # For Windows users: .\venv\Scripts\activate


Install the requirements:
The simulator uses graceful fallbacks for most heavy libraries, but for the full experience (PINNs, JIT acceleration, and RL), you'll want them all:
pip install -r requirements.txt


Getting Started
To launch the Digital Twin Console, run the main script:
python morphing-uav.py


Once it's running, open your browser to http://127.0.0.1:8050/. You can adjust morphing parameters like span, sweep, and twist in real-time and watch the physics engines update the plots instantly.
Data and Analysis
Research isn't much good without data. You can click the "Download Dataset" button directly in the dashboard to export a comprehensive CSV containing the current spanwise loading, bending moments, and aerodynamic coefficients for use in your own post-processing scripts.
How to Cite
If this simulator helps your research, please cite it using the record below:
@software{morphing_uav_simulator_2026,
  author       = Aaryan Chaulagain,
  title        = {Morphing UAV Simulator Suite: High-Fidelity Research Architecture},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19258496},
  url          = {[https://doi.org/10.5281/zenodo.19258496](https://doi.org/10.5281/zenodo.19258496)}
}


License
This work is licensed under a Creative Commons Attribution 4.0 International License (CC BY 4.0).
This license allows you to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator.
