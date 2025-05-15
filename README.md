# Langevin Simulations Optimizer

A Python library for running differentiable Langevin simulations of a Brownian
particle subjected to time-dependent potentials with trainable parameters. This
library enables optimization of equilibration time and work through
gradient-based methods. For details see:
D. Rengifo, G Téllez, [A machine learning approach to fast thermal equilibration](https://arxiv.org/abs/2504.08080)

## Features

- **Differentiable Simulations**: Built on PyTorch for end-to-end differentiable Langevin dynamics
- **Flexible Force Models**:
  - Harmonic forces with variable stiffness
  - Harmonic forces with variable center
  - Combined variable stiffness and center forces
  - Hermonic forces with stiffness and center given by polynomial functions of time.
- **Optimization Objectives**:
  - Equilibration time optimization
  - Work optimization
  - Custom loss functions
- **Training Utilities**:
  - Built-in training loops
  - Progress tracking
  - Visualization tools

## Installation

```bash
# Clone the repository
git clone https://github.com/GabrielTellez/langesim_optim.git
cd langesim_optim

# Install using Poetry
poetry install
```

## Quick Start

Here's a simple example of running a simulation with a variable stiffness harmonic force:

```python
import torch
from langesim_optim.simulator_forces import (
    Simulator,
    VariableStiffnessHarmonicForce
)
from langesim_optim.utilities import train_loop
from langesim_optim.loss_functions import loss_fn_k

# Create a force with variable stiffness
force = VariableStiffnessHarmonicForce(
    kappai=0.5,  # Initial stiffness
    kappaf=1.0,  # Final stiffness
    tf=0.1,      # Final time
    steps=100    # Number of interpolation points
)

# Create a simulator
sim = Simulator(
    dt=0.001,           # Time step
    tot_steps=1000,     # Total simulation steps
    noise_scaler=1.0,   # Noise strength
    force=force         # Force object
)

# Set up optimizer
optimizer = torch.optim.SGD(sim.parameters(), lr=0.001)

# The relaxation time of the system is 1/kf. 
# Before training, the system is not in thermal equilibrium at time tf

# Run training loop
losses = train_loop(
    epochs=100,
    sim=sim,
    tot_sims=1000,
    optimizer=optimizer,
    loss_fn=loss_fn_k,
    kf=1.0,
    ki=0.5
)

# After training, the protocol for the stiffness is such that 
# at time tf the system is now in thermal equilibrium.

```

## Documentation

For detailed documentation, visit our [documentation site](https://gabrieltellez.github.io/langesim_optim/).

## Features in Detail

### Force Models

The library provides several force models that can be used to simulate different physical scenarios:

- `VariableStiffnessHarmonicForce`: Harmonic oscillator with time-dependent stiffness
- `VariableCenterHarmonicForce`: Harmonic oscillator with time-dependent center
- `VariableStiffnessCenterHarmonicForce`: Combined variable stiffness and center
- `VariableStiffnessHarmonicForcePolynomial`: Polynomial-based stiffness protocol

### Loss Functions

The library provides loss functions for two main optimization objectives:

#### 1. Optimize Equilibration Time
These loss functions help find protocols that minimize the time needed for the system to reach equilibrium:

- `loss_fn_k`: Compares characteristic functions of final distributions to optimize equilibration time
- `loss_fn_variance`: Optimizes by comparing the final variance of particle positions
- `loss_fn_mean`: Optimizes by comparing the final average position of particles

#### 2. Optimize Work
These loss functions focus on minimizing the work done on the system:

- `loss_fn_work`: Optimizes the total work done on the system during the protocol
- `loss_fn_eq_work`: Optimizes both equilibration time and work simultaneously

### Training Utilities

The library includes utilities for training and visualization:

- `train_loop`: Main training loop with progress tracking
- `plot_test_hist`: Plot histograms of particle positions
- `plot_protocols`: Visualize force protocols

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
