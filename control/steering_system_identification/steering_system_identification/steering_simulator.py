import numpy as np
from acados_template import AcadosSimSolver, AcadosSim
from steering_model import export_steering_model

class SteeringSimulator:
    """Acados-based steering system simulator using ERK integration"""
    
    def __init__(self, tau=0.2, dt=0.1):
        self.tau = tau
        self.dt = dt
        
        # Create acados simulation object
        sim = AcadosSim()
        sim.model = export_steering_model()
        
        # Set integrator type and time step
        sim.solver_options.integrator_type = 'ERK'
        sim.solver_options.T = dt
        
        # Create integrator
        self.integrator = AcadosSimSolver(sim)
        
    def simulate_step(self, current_state, u_delayed, dt=None):
        """Simulate one step with ERK integration"""
        if dt is None:
            dt = self.dt
            
        # Update time step if different
        if abs(dt - self.dt) > 1e-6:
            self.integrator.set("T", dt)
            self.dt = dt
        
        # Set initial state
        self.integrator.set("x", current_state)
        
        # Set control input
        self.integrator.set("u", np.array([u_delayed]))
        
        # Integrate
        status = self.integrator.solve()
        
        if status != 0:
            print(f"Warning: Integrator failed with status {status}")
            # Fall back to simple Euler step
            steering, tau = current_state
            steering_dot = (-steering + u_delayed) / tau
            return np.array([steering + steering_dot * dt, tau])
        
        # Get next state
        return self.integrator.get("x")
    
    def simulate_trajectory(self, u_commands, initial_steering=0.0, tau=None, time_steps=None):
        """Simulate full trajectory with varying time steps"""
        if tau is None:
            tau = self.tau
            
        if time_steps is None:
            time_steps = np.full(len(u_commands), self.dt)
        
        # Initialize state
        state = np.array([initial_steering, tau])
        results = [initial_steering]
        
        for i, u_cmd in enumerate(u_commands[:-1]):  # Skip last command as we don't need to simulate beyond
            dt_current = time_steps[i] if i < len(time_steps) else self.dt
            
            # Apply delay - use command from delay_samples steps ago
            # For simulation, we'll assume delay is built into the u_commands
            u_delayed = u_cmd
            
            # Simulate one step
            state = self.simulate_step(state, u_delayed, dt_current)
            results.append(state[0])  # Store steering angle
            
        return np.array(results)

def simulate_steering_system(u_commands, tau, delay_samples=0, initial_steering=0.0, dt=0.1, time_steps=None):
    """Convenience function to simulate steering system with delay"""
    
    # Apply delay to commands
    u_delayed = np.zeros_like(u_commands)
    for i in range(len(u_commands)):
        if i >= delay_samples:
            u_delayed[i] = u_commands[i - delay_samples]
        else:
            u_delayed[i] = 0.0
    
    # Create simulator and run
    simulator = SteeringSimulator(tau=tau, dt=dt)
    return simulator.simulate_trajectory(u_delayed, initial_steering, tau, time_steps) 