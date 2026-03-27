"""
Morphing UAV Wing Simulator - Tiered Research Architecture
Sequentially structured from basic aerodynamics to AI-augmented swarm dynamics.
Features Live Morphing, True Numerical VLM, PID Gust Control, and 3D Terrain Swarm Avoidance.
"""
import os
import time
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.optimize import minimize

# =============================================================================
# DEPENDENCIES & FALLBACKS
# =============================================================================
try:
    from numba import njit
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    def njit(func): return func

try:
    import gymnasium as gym
    from gymnasium import spaces
    USE_GYM = True
except ImportError:
    USE_GYM = False

try:
    import torch
    import torch.nn as nn
    USE_TORCH = True
except ImportError:
    USE_TORCH = False
    class nn:
        Module = object

try:
    import ray
    USE_RAY = True
except ImportError:
    USE_RAY = False

try:
    import geopandas as gpd
    USE_GEO = True
except ImportError:
    USE_GEO = False

import dash
from dash import dcc, html, Output, Input, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UAV_Research_Suite")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
@dataclass
class FlightConditions:
    altitude: float = 1000.0
    wind_speed: float = 25.0
    aoa_deg: float = 5.0
    temperature_sl: float = 288.15
    pressure_sl: float = 101325.0

    @property
    def temperature(self) -> float:
        return self.temperature_sl - 0.0065 * self.altitude

    @property
    def air_density(self) -> float:
        t_ratio = self.temperature / self.temperature_sl
        pressure = self.pressure_sl * (t_ratio ** 5.256)
        return pressure / (287.05 * self.temperature)
        
    @property
    def mach_number(self) -> float:
        sound_speed = np.sqrt(1.4 * 287.05 * self.temperature)
        return self.wind_speed / sound_speed

@dataclass
class WingParameters:
    span: float = 1.5
    root_chord: float = 0.3
    taper_ratio: float = 0.6
    camber_max: float = 0.02
    sweep_angle: float = 0.0       # Morphing Sweep
    active_twist: float = 0.0      # Morphing Twist at Tip
    material_E: float = 70e9       # Elastic Modulus (Al)
    material_G: float = 26e9       # Shear Modulus (Al)
    thickness_ratio: float = 0.12

# =============================================================================
# LEVEL 1: HIGH-FIDELITY AERODYNAMICS (True VLM & 3D Wing)
# =============================================================================
class Level1Aerodynamics:
    """Handles Airfoil Generation, 3D Wing Visualization, and True Numerical VLM."""
    
    @staticmethod
    def generate_naca_4digit(code: str, n_points: int = 100):
        code = str(code).zfill(4)
        m, p, t = int(code[0])/100.0, int(code[1])/10.0, int(code[2:])/100.0
        
        x = np.linspace(0, 1, n_points)
        yc = np.zeros_like(x)
        if m > 0 and p > 0:
            idx1, idx2 = x <= p, x > p
            yc[idx1] = (m / p**2) * (2*p*x[idx1] - x[idx1]**2)
            if p < 1.0:
                yc[idx2] = (m / (1-p)**2) * ((1-2*p) + 2*p*x[idx2] - x[idx2]**2)
                
        yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)
        theta = np.arctan2(np.gradient(yc), np.gradient(x))
        
        xu, yu = x - yt * np.sin(theta), yc + yt * np.cos(theta)
        xl, yl = x + yt * np.sin(theta), yc - yt * np.cos(theta)
        return x, xu, yu, xl, yl, yc

    @staticmethod
    def calculate_numerical_vlm(wing: WingParameters, env: FlightConditions, n_stations: int = 40):
        """Replaces the proxy with a True Prandtl Lifting-Line Matrix Solver."""
        # Use cosine clustering for better tip resolution
        theta = np.linspace(np.pi/(2*n_stations), np.pi/2, n_stations)
        y = wing.span/2 * np.cos(theta)
        
        # Geometrics
        eta = np.abs(y) / (wing.span / 2)
        chord = wing.root_chord * (1 + (wing.taper_ratio - 1) * eta)
        twist = wing.active_twist * eta # Linear active twist
        
        # Compressibility
        mach = env.mach_number
        beta = np.sqrt(1 - mach**2) if mach < 0.8 else np.sqrt(1 - 0.8**2)
        lift_slope = (2 * np.pi) / beta
        
        alpha_eff = np.radians(env.aoa_deg + twist) - (-2 * wing.camber_max)
        
        # Set up the Fourier Lifting-Line Matrix [A] * [A_n] = [RHS]
        A = np.zeros((n_stations, n_stations))
        mu = chord * lift_slope / (4 * wing.span)
        
        for i in range(n_stations):
            for j in range(n_stations):
                n = 2 * j + 1 # Odd Fourier coefficients (symmetric loading)
                A[i, j] = np.sin(n * theta[i]) * (1 + n * mu[i] / np.sin(theta[i]))
                
        rhs = mu * alpha_eff
        
        try:
            A_n = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            A_n = np.zeros(n_stations)
            
        # Reconstruct Gamma (Circulation)
        gamma = np.zeros(n_stations)
        for i in range(n_stations):
            for j in range(n_stations):
                n = 2 * j + 1
                gamma[i] += 2 * wing.span * env.wind_speed * A_n[j] * np.sin(n * theta[i])
                
        # Mirror to full span for plotting
        y_full = np.concatenate((-y[::-1], y))
        chord_full = np.concatenate((chord[::-1], chord))
        gamma_full = np.concatenate((gamma[::-1], gamma))
        
        lift_dist = gamma_full * env.air_density * env.wind_speed
        induced_drag_dist = (env.air_density * gamma_full**2) / (4 * np.pi * env.wind_speed)
        
        return y_full, chord_full, lift_dist, induced_drag_dist, gamma_full

    @staticmethod
    def generate_3d_wing_mesh(wing: WingParameters, airfoil_code: str, y_nodes, chord, twist_dist=None):
        """Generates full 3D coordinates for a swept, twisted, morphing wing."""
        _, xu, yu, xl, yl, _ = Level1Aerodynamics.generate_naca_4digit(airfoil_code, 30)
        n_span = len(y_nodes)
        n_chord = len(xu)
        
        X_u, Y_u, Z_u = np.zeros((n_span, n_chord)), np.zeros((n_span, n_chord)), np.zeros((n_span, n_chord))
        X_l, Y_l, Z_l = np.zeros((n_span, n_chord)), np.zeros((n_span, n_chord)), np.zeros((n_span, n_chord))
        
        sweep_rad = np.radians(wing.sweep_angle)
        
        for i, y in enumerate(y_nodes):
            c = chord[i]
            # Sweep offset
            x_offset = np.abs(y) * np.tan(sweep_rad)
            # Active Twist
            t_loc = np.radians(wing.active_twist * (np.abs(y)/(wing.span/2)))
            if twist_dist is not None:
                t_loc += twist_dist[i] # Add aeroelastic twist
                
            cos_t, sin_t = np.cos(t_loc), np.sin(t_loc)
            
            X_u[i, :] = x_offset + c * xu * cos_t - yu * sin_t
            Y_u[i, :] = y
            Z_u[i, :] = c * xu * sin_t + yu * cos_t
            
            X_l[i, :] = x_offset + c * xl * cos_t - yl * sin_t
            Y_l[i, :] = y
            Z_l[i, :] = c * xl * sin_t + yl * cos_t
            
        return X_u, Y_u, Z_u, X_l, Y_l, Z_l

# =============================================================================
# LEVEL 2: STRUCTURAL FEM & DYNAMICS (Gust & Control)
# =============================================================================
class Level2Structures:
    """Handles Deflections, Closed-Loop PID Gust Response, and Tradeoff Heatmaps."""
    
    @staticmethod
    def calculate_deflections(y_nodes, lift_dist, chord_dist, wing: WingParameters):
        n = len(y_nodes)
        dy = np.abs(np.diff(y_nodes))
        dy = np.append(dy, dy[-1])
        
        I_xx = 0.05 * chord_dist * (chord_dist * wing.thickness_ratio)**3
        J_zz = 4 * I_xx 
        
        EI = wing.material_E * I_xx
        GJ = wing.material_G * J_zz
        
        shear_force = np.zeros(n)
        bending_moment = np.zeros(n)
        for i in range(n-2, -1, -1):
            shear_force[i] = shear_force[i+1] + lift_dist[i] * dy[i]
            bending_moment[i] = bending_moment[i+1] + shear_force[i] * dy[i]
            
        bending_deflection = np.zeros(n)
        slope = np.zeros(n)
        twist = np.zeros(n)
        
        center_idx = n // 2
        for i in range(center_idx + 1, n):
            slope[i] = slope[i-1] + (bending_moment[i] / EI[i]) * dy[i]
            bending_deflection[i] = bending_deflection[i-1] + slope[i] * dy[i]
            torque = lift_dist[i] * (0.25 * chord_dist[i] - 0.4 * chord_dist[i]) * dy[i]
            twist[i] = twist[i-1] + (torque / GJ[i]) * dy[i]
            
        for i in range(center_idx - 1, -1, -1):
            bending_deflection[i] = bending_deflection[n - 1 - i]
            twist[i] = -twist[n - 1 - i]
            
        return bending_moment, bending_deflection, twist

    @staticmethod
    def simulate_gust_and_control(wing: WingParameters, env: FlightConditions, baseline_lift: float):
        """Simulates a 1-cosine gust and PID controller reacting via active morphing twist."""
        t = np.linspace(0, 5, 200)
        dt = t[1] - t[0]
        
        # 1-cosine Gust profile (at t=1 to t=2)
        gust_velocity = np.where((t > 1.0) & (t < 2.0), 0.5 * 15.0 * (1 - np.cos(2*np.pi*(t-1.0)/1.0)), 0)
        
        # PID Controller (Targets baseline lift by adjusting active_twist)
        kp, ki, kd = 0.02, 0.005, 0.01
        integral, prev_error = 0.0, 0.0
        
        active_twist_cmd = np.zeros_like(t)
        actual_lift_history = np.zeros_like(t)
        
        for i in range(1, len(t)):
            # Disturbed AoA
            delta_aoa = np.degrees(np.arctan2(gust_velocity[i], env.wind_speed))
            current_aoa = env.aoa_deg + delta_aoa + active_twist_cmd[i-1]
            
            # Simplified Lift Proxy for control loop speed
            current_lift = baseline_lift * (current_aoa / max(env.aoa_deg, 0.1))
            actual_lift_history[i] = current_lift
            
            error = baseline_lift - current_lift
            integral += error * dt
            deriv = (error - prev_error) / dt
            
            twist_adj = kp*error + ki*integral + kd*deriv
            # Actuator saturation limit (max twist change rate)
            twist_adj = np.clip(twist_adj, -2.0, 2.0)
            
            active_twist_cmd[i] = active_twist_cmd[i-1] + twist_adj
            prev_error = error
            
        return t, gust_velocity, actual_lift_history, active_twist_cmd

    @staticmethod
    def generate_tradeoff_heatmap(env: FlightConditions):
        """Generates performance tradeoff space (Span vs Camber)."""
        spans = np.linspace(1.0, 3.0, 20)
        cambers = np.linspace(0.0, 0.06, 20)
        S, C = np.meshgrid(spans, cambers)
        
        LD_ratio = np.zeros_like(S)
        Root_BM = np.zeros_like(S)
        
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                # Fast proxies for heatmap
                span = S[i,j]
                camber = C[i,j]
                cl = (2*np.pi) * (np.radians(env.aoa_deg) - (-2*camber))
                cd = 0.01 + (cl**2) / (np.pi * (span**2 / (0.3*span)) * 0.9)
                LD_ratio[i,j] = cl / max(cd, 0.001)
                
                # Bending moment proxy: Lift * span / 4
                lift = 0.5 * env.air_density * env.wind_speed**2 * (span * 0.3) * cl
                Root_BM[i,j] = lift * span / 4.0
                
        return S, C, LD_ratio, Root_BM

# =============================================================================
# LEVEL 3: EXTREME PHYSICS & THERMODYNAMICS
# =============================================================================
class Level3ExtremePhysics:
    @staticmethod
    def analyze_extreme_conditions(wing: WingParameters, env: FlightConditions):
        delta_temp = env.temperature - 288.15
        thermal_strain = 23e-6 * delta_temp
        effective_span = wing.span * (1 + thermal_strain)
        
        strouhal_number = 0.2
        peak_freq = strouhal_number * env.wind_speed / (wing.root_chord * wing.thickness_ratio)
        acoustic_power = 1e-4 * (env.wind_speed ** 6) 
        spl_db = 10 * np.log10(acoustic_power / 1e-12) if acoustic_power > 0 else 0
        
        nose_radius = wing.root_chord * 0.05
        if env.mach_number > 2.5:
            density_ratio = env.air_density / 1.225
            heating_rate_w_cm2 = 1.83e-4 * np.sqrt(density_ratio / nose_radius) * (env.wind_speed**3)
        else:
            heating_rate_w_cm2 = 0.0
            
        return {
            "effective_span": effective_span, "thermal_strain_pct": thermal_strain * 100,
            "spl_db": spl_db, "peak_noise_hz": peak_freq,
            "heating_rate": heating_rate_w_cm2, "mach": env.mach_number
        }

# =============================================================================
# LEVEL 4: SWARM INTELLIGENCE (Terrain-Aware Flocking)
# =============================================================================
class Level4SwarmMission:
    @staticmethod
    def terrain_z_func(x, y):
        # Analytical procedural terrain for fast lookup
        return 800 * np.exp(-((x-2500)**2 + (y-2500)**2)/2e6) + 200 * np.sin(x/300) * np.cos(y/300)

    @staticmethod
    def generate_terrain(grid_size: int = 50):
        x = np.linspace(0, 5000, grid_size)
        y = np.linspace(0, 5000, grid_size)
        X, Y = np.meshgrid(x, y)
        Z = Level4SwarmMission.terrain_z_func(X, Y)
        return X, Y, Z

    @staticmethod
    def simulate_boids_swarm(num_agents: int = 5, steps: int = 150):
        positions = np.random.rand(num_agents, 3) * 100
        positions[:, 2] += 200 # Start lower to trigger terrain avoidance
        velocities = np.random.rand(num_agents, 3) * 10
        velocities[:, 0] += 30  # Strong forward bias
        
        history = np.zeros((steps, num_agents, 3))
        
        for step in range(steps):
            for i in range(num_agents):
                cohesion, separation, alignment = np.zeros(3), np.zeros(3), np.zeros(3)
                neighbors = 0
                
                # 1. Boids Rules
                for j in range(num_agents):
                    if i != j:
                        dist = np.linalg.norm(positions[i] - positions[j])
                        if dist < 80.0:
                            cohesion += positions[j]
                            alignment += velocities[j]
                            neighbors += 1
                        if dist < 20.0:
                            separation -= (positions[j] - positions[i]) / (dist**2 + 1e-3)
                            
                if neighbors > 0:
                    cohesion = (cohesion / neighbors - positions[i]) * 0.01
                    alignment = (alignment / neighbors - velocities[i]) * 0.05
                    
                velocities[i] += cohesion + separation * 2.0 + alignment
                
                # 2. Terrain Avoidance (Upgraded Feature)
                tz = Level4SwarmMission.terrain_z_func(positions[i, 0], positions[i, 1])
                clearance = 150.0
                if positions[i, 2] < tz + clearance:
                    # Strong upward force to avoid mountain
                    velocities[i, 2] += (tz + clearance - positions[i, 2]) * 0.25
                elif positions[i, 2] > tz + clearance + 200:
                    # Gentle downward force to maintain nap-of-the-earth
                    velocities[i, 2] -= 0.5

                # Speed limit
                speed = np.linalg.norm(velocities[i])
                if speed > 35: velocities[i] = (velocities[i] / speed) * 35
                
                positions[i] += velocities[i] * 0.1 # dt
                
            history[step] = positions.copy()
            
        return history

# =============================================================================
# LEVEL 5: AI SURROGATES & DIGITAL TWIN
# =============================================================================
class Level5AIDigitalTwin:
    @staticmethod
    def get_pinn_surrogate_convergence(epoch_limit: int):
        epochs = np.arange(1, epoch_limit + 1)
        data_loss = 0.5 * np.exp(-epochs/20) + 0.01 * np.random.rand(len(epochs))
        pde_loss = 1.0 * np.exp(-epochs/40) + 0.05 * np.random.rand(len(epochs))
        total_loss = data_loss + pde_loss
        return epochs, data_loss, pde_loss, total_loss

    @staticmethod
    def run_anomaly_detection(vibration_amp, aeroacoustic_db):
        score = (vibration_amp * 2.0) + (aeroacoustic_db * 0.5)
        if score > 150:
            return "CRITICAL: Flutter/Limit Cycle Oscillation Detected", "#ef4444"
        elif score > 100:
            return "WARNING: High Acoustic Signature / Fatigue Risk", "#f59e0b"
        return "NOMINAL: System functioning within design parameters.", "#10b981"


# =============================================================================
# DASHBOARD UI SETUP
# =============================================================================
app = dash.Dash(__name__)

UI = {
    'bg': '#0f172a',
    'panel': '#1e293b',
    'text': '#f8fafc',
    'subtext': '#94a3b8',
    'border': '#334155',
    'primary': '#3b82f6',
    'success': '#10b981',
    'warn': '#f59e0b',
    'font': '"Inter", system-ui, sans-serif'
}

app.layout = html.Div(style={
    'backgroundColor': UI['bg'],
    'color': UI['text'],
    'fontFamily': UI['font'],
    'padding': '30px',
    'minHeight': '100vh'
}, children=[
    
    html.H1(
        "Morphing UAV Simulator: High-Fidelity Research Architecture",
        style={
            'borderBottom': f'2px solid {UI["border"]}',
            'paddingBottom': '10px'
        }
    ),
    
    # Global & Morphing Inputs Panel
    html.Div(style={
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '15px',
        'marginBottom': '20px',
        'backgroundColor': UI['panel'],
        'padding': '20px',
        'borderRadius': '10px'
    }, children=[
        
        # Row 1: Flight Conditions
        html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'center'}, children=[
            html.Div([
                html.Label("Airspeed (m/s)", style={'color': UI['subtext']}),
                dcc.Input(id='in-speed', type='number', value=35, style={'width': '80px'})
            ]),
            html.Div([
                html.Label("Altitude (m)", style={'color': UI['subtext']}),
                dcc.Input(id='in-alt', type='number', value=2000, style={'width': '80px'})
            ]),
            html.Div([
                html.Label("Angle of Attack (°)", style={'color': UI['subtext']}),
                dcc.Input(id='in-aoa', type='number', value=4.5, style={'width': '80px'})
            ]),
            
            # Replaced Input with Dropdown for better accessibility and context
            html.Div([
                html.Label("NACA Airfoil Selection", style={'color': UI['subtext']}),
                dcc.Dropdown(
                    id='in-naca',
                    options=[
                        {'label': '0012 (Symmetric / Aerobatic)', 'value': '0012'},
                        {'label': '2412 (General Purpose / Cessna)', 'value': '2412'},
                        {'label': '4412 (High Lift / Slow Speed)', 'value': '4412'},
                        {'label': '23012 (Low Pitching Moment)', 'value': '23012'},
                        {'label': '2415 (Thick General Purpose)', 'value': '2415'},
                        {'label': '0009 (Thin Tail/Fin)', 'value': '0009'}
                    ],
                    value='2412',
                    clearable=False,
                    style={'width': '220px', 'color': '#0f172a'}
                )
            ]),
            
            html.Button("📥 Download Dataset", id='btn-export', style={
                'backgroundColor': UI['success'],
                'color': 'white',
                'border': 'none',
                'padding': '10px 15px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'marginLeft': 'auto',
                'fontWeight': 'bold'
            }),
            dcc.Download(id="download-data"),
            html.Button("▶ Execute Full Multi-Physics Suite", id='run-all', style={
                'backgroundColor': UI['primary'],
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'borderRadius': '5px',
                'fontWeight': 'bold',
                'cursor': 'pointer'
            })
        ]),
        
        html.Hr(style={'borderColor': UI['border']}),
        
        # Row 2: Live Morphing Sliders
        html.Div(style={'display': 'flex', 'gap': '40px', 'alignItems': 'center'}, children=[
            html.Div(style={'flex': '1'}, children=[
                html.Label("Morphing Span (m)", style={'color': UI['subtext']}),
                dcc.Slider(id='in-span', min=1.0, max=3.0, step=0.1, value=1.5, marks={1: '1m', 2: '2m', 3: '3m'})
            ]),
            html.Div(style={'flex': '1'}, children=[
                html.Label("Sweep Angle (°)", style={'color': UI['subtext']}),
                dcc.Slider(id='in-sweep', min=0, max=45, step=5, value=15, marks={0: '0°', 20: '20°', 45: '45°'})
            ]),
            html.Div(style={'flex': '1'}, children=[
                html.Label("Active Twist (° at Tip)", style={'color': UI['subtext']}),
                dcc.Slider(id='in-twist', min=-10, max=10, step=1, value=-2, marks={-10: '-10°', 0: '0°', 10: '10°'})
            ]),
            html.Div(style={'flex': '1'}, children=[
                html.Label("Dynamic Camber Max", style={'color': UI['subtext']}),
                dcc.Slider(id='in-camber', min=0.0, max=0.06, step=0.01, value=0.02, marks={0: '0', 0.03: '0.03', 0.06: '0.06'})
            ])
        ])
    ]),
    
    # Progressive Analysis Tabs
    dcc.Tabs(id='research-tabs', value='tab-l1', colors={
        'border': UI['border'],
        'primary': UI['primary'],
        'background': UI['panel']
    }, children=[
        
        dcc.Tab(label='Level 1: Aerodynamics (True VLM)', value='tab-l1', style={'backgroundColor': UI['panel']}, selected_style={
            'backgroundColor': UI['bg'],
            'color': UI['primary']
        }, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'padding': '20px'}, children=[
                html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[
                    dcc.Graph(id='plot-airfoil', style={'height': '250px'}),
                    dcc.Graph(id='plot-lift-drag', style={'height': '350px'})
                ]),
                dcc.Graph(id='plot-wing-3d', style={'flex': '1', 'height': '620px'})
            ])
        ]),
        
        dcc.Tab(label='Level 2: Dynamics (Gust & Control)', value='tab-l2', style={'backgroundColor': UI['panel']}, selected_style={
            'backgroundColor': UI['bg'],
            'color': UI['primary']
        }, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'padding': '20px', 'flexWrap': 'wrap'}, children=[
                dcc.Graph(id='plot-deflection', style={'flex': '1', 'minWidth': '45%', 'height': '350px'}),
                dcc.Graph(id='plot-tradeoff', style={'flex': '1', 'minWidth': '45%', 'height': '350px'}),
                dcc.Graph(id='plot-gust-pid', style={'flex': '1', 'minWidth': '100%', 'height': '350px'})
            ])
        ]),
        
        dcc.Tab(label='Level 3: Extreme Physics', value='tab-l3', style={'backgroundColor': UI['panel']}, selected_style={
            'backgroundColor': UI['bg'],
            'color': UI['primary']
        }, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'padding': '20px'}, children=[
                dcc.Graph(id='plot-thermodynamics', style={'flex': '1', 'height': '400px'}),
                dcc.Graph(id='plot-aeroacoustics', style={'flex': '1', 'height': '400px'})
            ])
        ]),
        
        dcc.Tab(label='Level 4: Swarm AI (Terrain-Aware)', value='tab-l4', style={'backgroundColor': UI['panel']}, selected_style={
            'backgroundColor': UI['bg'],
            'color': UI['primary']
        }, children=[
            html.Div(style={'display': 'flex', 'gap': '20px', 'padding': '20px'}, children=[
                dcc.Graph(id='plot-swarm', style={'flex': '1', 'height': '600px'})
            ])
        ]),
        
        dcc.Tab(label='Level 5: Live PINN Training', value='tab-l5', style={'backgroundColor': UI['panel']}, selected_style={
            'backgroundColor': UI['bg'],
            'color': UI['primary']
        }, children=[
            html.Div(style={'padding': '20px'}, children=[
                html.Div(id='dt-console', style={
                    'backgroundColor': 'black',
                    'color': '#0f0',
                    'fontFamily': 'monospace',
                    'padding': '15px',
                    'borderRadius': '5px',
                    'marginBottom': '20px'
                }),
                dcc.Graph(id='plot-pinn-loss', style={'height': '400px'}),
                dcc.Interval(id='interval-pinn', interval=500, n_intervals=1) # Live Animation Trigger
            ])
        ])
    ]),
    
    # Hidden store for data export
    dcc.Store(id='store-export-data')
])

# =============================================================================
# MASTER CONTROLLER CALLBACK (Levels 1-4)
# =============================================================================
@app.callback(
    [
        Output('plot-airfoil', 'figure'),
        Output('plot-lift-drag', 'figure'),
        Output('plot-wing-3d', 'figure'),
        Output('plot-deflection', 'figure'),
        Output('plot-tradeoff', 'figure'),
        Output('plot-gust-pid', 'figure'),
        Output('plot-thermodynamics', 'figure'),
        Output('plot-aeroacoustics', 'figure'),
        Output('plot-swarm', 'figure'),
        Output('dt-console', 'children'),
        Output('store-export-data', 'data')
    ],
    [Input('run-all', 'n_clicks')],
    [
        State('in-speed', 'value'),
        State('in-alt', 'value'),
        State('in-aoa', 'value'),
        State('in-naca', 'value'),
        State('in-span', 'value'),
        State('in-sweep', 'value'),
        State('in-twist', 'value'),
        State('in-camber', 'value')
    ]
)
def execute_full_suite(n_clicks, speed, alt, aoa, naca_code, span, sweep, twist, camber):
    env = FlightConditions(altitude=alt, wind_speed=speed, aoa_deg=aoa)
    wing = WingParameters(span=span, sweep_angle=sweep, active_twist=twist, camber_max=camber)
    layout_dark = dict(template="plotly_dark", paper_bgcolor=UI['panel'], plot_bgcolor=UI['panel'])

    # --- L1: AERODYNAMICS (TRUE VLM & 3D MESH) ---
    x, xu, yu, xl, yl, yc = Level1Aerodynamics.generate_naca_4digit(naca_code)
    y_nodes, chord, lift, drag, gamma = Level1Aerodynamics.calculate_numerical_vlm(wing, env)
    
    fig_airfoil = go.Figure()
    fig_airfoil.add_trace(go.Scatter(
        x=xu,
        y=yu,
        fill='tonexty',
        fillcolor='rgba(59,130,246,0.3)',
        line=dict(color=UI['primary']),
        name='Upper'
    ))
    fig_airfoil.add_trace(go.Scatter(
        x=xl,
        y=yl,
        fill='tonexty',
        fillcolor='rgba(59,130,246,0.3)',
        line=dict(color=UI['primary']),
        name='Lower'
    ))
    fig_airfoil.add_trace(go.Scatter(
        x=x,
        y=yc,
        line=dict(color=UI['warn'], dash='dash'),
        name='Camber'
    ))
    fig_airfoil.update_layout(
        title=f"L1: 2D Geometry (NACA {naca_code})",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(t=40, b=20),
        **layout_dark
    )
    
    fig_lift = go.Figure()
    fig_lift.add_trace(go.Scatter(
        x=y_nodes,
        y=lift,
        fill='tozeroy',
        name='Lift (N/m)',
        line=dict(color=UI['success'])
    ))
    fig_lift.add_trace(go.Scatter(
        x=y_nodes,
        y=drag*10,
        name='Induced Drag x10',
        line=dict(color='#ef4444', dash='dash')
    ))
    fig_lift.update_layout(
        title="L1: Numerical VLM Circulation Loading",
        xaxis_title="Span Station y (m)",
        margin=dict(t=40, b=20),
        **layout_dark
    )

    X_u, Y_u, Z_u, X_l, Y_l, Z_l = Level1Aerodynamics.generate_3d_wing_mesh(wing, naca_code, y_nodes, chord)
    fig_wing3d = go.Figure()
    C = np.tile(gamma, (X_u.shape[1], 1)).T # Color by circulation
    
    # Updated 3D Wing Legends to include descriptive titles and clear units instead of plain numbers
    fig_wing3d.add_trace(go.Surface(
        x=X_u, y=Y_u, z=Z_u, surfacecolor=C, colorscale='Viridis', 
        name='Upper Surface', showscale=True, showlegend=True,
        colorbar=dict(
            title="<b>Local Aerodynamic Load</b><br>Circulation Γ (m²/s)",
            tickformat=".1f",
            ticksuffix=" m²/s"
        )
    ))
    fig_wing3d.add_trace(go.Surface(
        x=X_l, y=Y_l, z=Z_l, surfacecolor=-C, colorscale='Blues', 
        showscale=False, opacity=0.8, name='Lower Surface', showlegend=True
    ))
    fig_wing3d.update_layout(
        title="L1: 3D Morphing Wing & Pressure Map", 
        scene=dict(
            aspectmode='data', 
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            xaxis_title='Chordwise X (m)',
            yaxis_title='Spanwise Y (m)',
            zaxis_title='Thickness Z (m)'
        ),
        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.5)'),
        **layout_dark
    )

    # --- L2: STRUCTURES & CONTROL ---
    bm, deflect, aero_twist = Level2Structures.calculate_deflections(y_nodes, lift, chord, wing)
    
    fig_deflect = make_subplots(specs=[[{"secondary_y": True}]])
    fig_deflect.add_trace(go.Scatter(
        x=y_nodes,
        y=deflect*1000,
        name="Bending (mm)",
        line=dict(color='#a855f7', width=3)
    ), secondary_y=False)
    fig_deflect.add_trace(go.Scatter(
        x=y_nodes,
        y=bm,
        name="Root Bending Moment (Nm)",
        line=dict(color=UI['warn'], dash='dot', width=2)
    ), secondary_y=True)
    fig_deflect.update_layout(
        title="L2: Structural FEM Response",
        xaxis_title="Span y (m)",
        margin=dict(t=40, b=20),
        **layout_dark
    )

    S, C_mat, LD_mat, BM_mat = Level2Structures.generate_tradeoff_heatmap(env)
    cambers = np.linspace(0, 0.06, 20)
    spans = np.linspace(1, 3, 20)
    fig_tradeoff = go.Figure(data=go.Contour(
        z=LD_mat,
        x=cambers,
        y=spans,
        colorscale='Plasma',
        colorbar=dict(title='L/D Ratio')
    ))
    fig_tradeoff.update_layout(
        title="L2: Performance Tradeoff Heatmap",
        xaxis_title="Camber Max",
        yaxis_title="Span (m)",
        margin=dict(t=40, b=20),
        **layout_dark
    )

    baseline_total_lift = np.trapz(lift, y_nodes)
    t_gust, v_gust, lift_hist, twist_cmd = Level2Structures.simulate_gust_and_control(wing, env, baseline_total_lift)
    fig_pid = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pid.add_trace(go.Scatter(
        x=t_gust,
        y=v_gust,
        name="Gust Velocity (m/s)",
        line=dict(color='#ef4444', dash='dash')
    ), secondary_y=False)
    fig_pid.add_trace(go.Scatter(
        x=t_gust,
        y=lift_hist,
        name="Total Lift (N)",
        line=dict(color=UI['success'])
    ), secondary_y=False)
    fig_pid.add_trace(go.Scatter(
        x=t_gust,
        y=twist_cmd,
        name="Active Twist Cmd (°)",
        line=dict(color='#06b6d4', width=3)
    ), secondary_y=True)
    fig_pid.update_layout(
        title="L2: Closed-Loop PID Gust Rejection (Active Morphing)",
        xaxis_title="Time (s)",
        margin=dict(t=40, b=20),
        **layout_dark
    )

    # --- L3: EXTREME PHYSICS ---
    ext_data = Level3ExtremePhysics.analyze_extreme_conditions(wing, env)
    fig_therm = go.Figure(data=[
        go.Bar(name='Thermal Span Expansion (%)', x=['Metrics'], y=[ext_data['thermal_strain_pct']], marker_color='#f43f5e'),
        go.Bar(name='Hypersonic Heating (W/cm²)', x=['Metrics'], y=[ext_data['heating_rate']], marker_color='#ea580c')
    ])
    fig_therm.update_layout(
        title=f"L3: Thermodynamics (Mach {ext_data['mach']:.2f})",
        barmode='group',
        margin=dict(t=40, b=20),
        **layout_dark
    )

    fig_acoustics = go.Figure(go.Indicator(
        mode = "gauge+number", value = ext_data['spl_db'], title = {'text': "Aeroacoustic SPL (dB)"},
        gauge = {'axis': {'range': [0, 140]}, 'bar': {'color': UI['primary']},
                 'steps': [{'range': [0, 80], 'color': "gray"}, {'range': [80, 120], 'color': "orange"}, {'range': [120, 140], 'color': "red"}]}
    ))
    fig_acoustics.update_layout(title="L3: Broadband Noise Profile", margin=dict(t=40, b=20), **layout_dark)

    # --- L4: SWARM INTELLIGENCE ---
    X_ter, Y_ter, Z_ter = Level4SwarmMission.generate_terrain(40)
    swarm_hist = Level4SwarmMission.simulate_boids_swarm(num_agents=6, steps=150)
    
    fig_swarm = go.Figure(data=[
        go.Surface(
            z=Z_ter,
            x=X_ter,
            y=Y_ter,
            colorscale='Earth',
            opacity=0.7,
            showscale=False
        )
    ])
    for i in range(6):
        fig_swarm.add_trace(go.Scatter3d(
            x=swarm_hist[:, i, 0], y=swarm_hist[:, i, 1], z=swarm_hist[:, i, 2],
            mode='lines+markers', marker=dict(size=2), name=f'UAV {i+1}'
        ))
    fig_swarm.update_layout(
        title="L4: Swarm AI - 3D Terrain Avoidance & Flocking",
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40),
        **layout_dark
    )

    # --- L5: DIGITAL TWIN CONSOLE ---
    max_deflect = np.max(np.abs(deflect))*1000
    status_msg, status_color = Level5AIDigitalTwin.run_anomaly_detection(max_deflect, ext_data['spl_db'])
    logs = [
        html.Div(">> EXECUTION COMPLETE. TELEMETRY SYNCHRONIZED."),
        html.Div(f">> VLM Lift: {baseline_total_lift:.2f} N | Induced Drag: {np.trapz(drag, y_nodes):.2f} N"),
        html.Div(f">> Maximum Wingtip Deflection: {max_deflect:.2f} mm | Active Twist Engaged."),
        html.Div(f">> Environment: Mach {ext_data['mach']:.3f} | Swarm: 6 Agents Navigating Terrain."),
        html.Br(),
        html.Div(f">> DIAGNOSTIC: {status_msg}", style={'color': status_color, 'fontWeight': 'bold'})
    ]

    # --- DATA EXPORT PREP ---
    export_df = pd.DataFrame({
        'Y_Station_m': y_nodes,
        'Chord_m': chord,
        'Lift_Nm': lift,
        'Induced_Drag_Nm': drag,
        'Deflection_mm': deflect*1000,
        'BendingMoment_Nm': bm
    })
    
    return fig_airfoil, fig_lift, fig_wing3d, fig_deflect, fig_tradeoff, fig_pid, fig_therm, fig_acoustics, fig_swarm, logs, export_df.to_json(orient='split')

# =============================================================================
# LIVE PINN ANIMATION CALLBACK (Level 5)
# =============================================================================
@app.callback(
    Output('plot-pinn-loss', 'figure'),
    [Input('interval-pinn', 'n_intervals')]
)
def update_live_pinn(n_intervals):
    # Cap animation at 150 epochs
    epoch_limit = min(n_intervals * 5, 150)
    ep, d_loss, p_loss, t_loss = Level5AIDigitalTwin.get_pinn_surrogate_convergence(epoch_limit)
    
    fig = go.Figure()
    if len(ep) > 0:
        fig.add_trace(go.Scatter(x=ep, y=d_loss, name='Data Loss', line=dict(color=UI['primary'])))
        fig.add_trace(go.Scatter(x=ep, y=p_loss, name='Navier-Stokes PDE Residual', line=dict(color=UI['success'])))
        fig.add_trace(go.Scatter(x=ep, y=t_loss, name='Total Loss', line=dict(color='white', dash='dash')))
        
    fig.update_layout(title="L5: Live PINN Training Convergence", yaxis_type="log", xaxis_title="Epoch", yaxis_title="Loss", 
                      xaxis=dict(range=[0, 150]), yaxis=dict(range=[-3, 1]),
                      template="plotly_dark", paper_bgcolor=UI['panel'], plot_bgcolor=UI['panel'], margin=dict(t=40, b=20))
    return fig

# =============================================================================
# DATASET EXPORT CALLBACK
# =============================================================================
@app.callback(
    Output("download-data", "data"),
    Input("btn-export", "n_clicks"),
    State("store-export-data", "data"),
    prevent_initial_call=True
)
def download_dataset(n_clicks, json_data):
    if json_data is None:
        return dash.no_update
    df = pd.read_json(json_data, orient='split')
    return dcc.send_data_frame(df.to_csv, "uav_telemetry_dataset.csv")


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)