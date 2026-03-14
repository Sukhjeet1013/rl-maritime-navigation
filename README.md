# Autonomous Maritime Navigation using Reinforcement Learning

This project demonstrates an autonomous ship navigation system trained
using **Reinforcement Learning (PPO)**.\
The agent learns to navigate through a dynamic ocean environment while
avoiding moving obstacles and ocean current disturbances.

------------------------------------------------------------------------

# Project Overview

Autonomous maritime navigation is a challenging problem involving
dynamic environments, unpredictable disturbances, and collision
avoidance.

In this project, a **custom reinforcement learning environment** was
built using **Gymnasium**. The agent learns to navigate a ship across a
grid-based ocean environment while:

-   Avoiding moving obstacles
-   Handling ocean current disturbances
-   Navigating toward randomly generated goal locations

The agent is trained using **Proximal Policy Optimization (PPO)**
implemented with **Stable-Baselines3**.

------------------------------------------------------------------------

# Environment Design

## Environment Features

-   **Grid Size:** 10 × 10 ocean map
-   **Agent:** Autonomous ship
-   **Goal:** Reach a target location
-   **Action Space:** 4 discrete actions
    -   Move Up\
    -   Move Down\
    -   Move Left\
    -   Move Right

## Dynamic Challenges

The environment simulates several realistic navigation challenges:

-   Moving obstacles
-   Random obstacle direction changes
-   Ocean current disturbances
-   Randomized goal positions
-   Time-limited navigation episodes

These conditions create a **stochastic navigation environment**, making
the task more complex than simple pathfinding.

------------------------------------------------------------------------

# Reinforcement Learning Algorithm

The agent is trained using:

**Proximal Policy Optimization (PPO)**

### Training Configuration

-   **Algorithm:** PPO\
-   **Library:** Stable-Baselines3\
-   **Policy Network:** MLP\
-   **Training Steps:** 4,000,000\
-   **Observation Space:** 20 features\
-   **Action Space:** 4 discrete actions

------------------------------------------------------------------------

# State Representation

The agent receives a **20-dimensional observation vector** including:

-   Ship position
-   Goal position
-   Relative distance to the goal
-   Obstacle positions

All observations are **normalized between -1 and 1** to improve training
stability.

------------------------------------------------------------------------

# Training

Training is performed using the custom Gymnasium environment.

### Run Training

``` bash
python -m training.train_agent
```

### Training Time

\~20--30 minutes on CPU.

------------------------------------------------------------------------

# Visualization

The trained agent can be visualized using **Pygame**, showing:

-   Ship navigation
-   Moving obstacles
-   Ocean current direction

Run the simulation:

``` bash
python render/run_simulation.py
```

------------------------------------------------------------------------

# Performance Analysis

Training performance and evaluation results:

![Training Results](image-1.png)

![Goal Success Rate](image-2.png)

![Crash Analysis](image-3.png)

------------------------------------------------------------------------

# Installation

Install the required dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Running the Project

### Train the RL Agent

``` bash
python -m training.train_agent
```

### Run the Simulation

``` bash
python render/run_simulation.py
```

------------------------------------------------------------------------

# Technologies Used

-   **Python**
-   **Gymnasium** -- Reinforcement learning environment framework
-   **Stable-Baselines3** -- Reinforcement learning algorithms
-   **NumPy** -- Numerical computation
-   **Pygame** -- Simulation visualization
-   **Matplotlib** -- Training and performance visualization

------------------------------------------------------------------------

# Key Learning Outcomes

This project demonstrates:

-   Designing **custom reinforcement learning environments**
-   Training **PPO agents in stochastic navigation environments**
-   Implementing **reward shaping strategies for navigation tasks**
-   Visualizing **reinforcement learning policies in a simulated
    environment**

------------------------------------------------------------------------

# Future Improvements

Potential extensions include:

-   Continuous control navigation
-   Sensor-based obstacle detection
-   Multi-agent maritime traffic simulation
-   Deep reinforcement learning with **CNN-based observations**

------------------------------------------------------------------------

# Author

**Sukhjeet Singh**\
AI / Machine Learning Enthusiast
