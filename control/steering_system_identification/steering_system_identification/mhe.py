import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from casadi import SX, vertcat
from scipy.linalg import block_diag
from collections import deque
from steering_simulator import simulate_steering_system

def export_steering_mhe_model():
    """Export steering system MHE model with time constant as augmented state"""
    
    model_name = 'steering_mhe_model'
    
    # States: [steering_angle, time_constant]
    # Following the pendulum example pattern where parameter becomes an augmented state
    steering = SX.sym('steering')
    tau = SX.sym('tau')  # time constant as augmented state
    x = vertcat(steering, tau)
    
    # Process noise (control input in MHE)
    w_steering = SX.sym('w_steering')  # process noise for steering dynamics
    w = w_steering  # Only steering has process noise, tau is parameter-like
    
    # State derivatives
    steering_dot = SX.sym('steering_dot')
    tau_dot = SX.sym('tau_dot')
    xdot = vertcat(steering_dot, tau_dot)
    
    # Parameters (delayed commanded steering)
    u_delayed = SX.sym('u_delayed')  # delayed commanded steering
    p = u_delayed
    
    # First-order dynamics with delay already incorporated in the input
    # d(steering)/dt = (-steering + u_delayed) / tau + w_steering
    # d(tau)/dt = 0 (parameter evolves very slowly, almost constant)
    f_expl = vertcat(
        (-steering + u_delayed) / tau + w_steering,
        0  # tau is nearly constant (parameter)
    )
    
    f_impl = xdot - f_expl
    
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = w  # process noise
    model.p = p  # delayed commanded steering
    model.name = model_name
    
    return model

def export_steering_mhe_solver(model, N, h, Q, Q0, R):
    """Export MHE solver following acados pendulum example pattern"""
    
    ocp_mhe = AcadosOcp()
    ocp_mhe.model = model
    
    nx_augmented = model.x.rows()  # 2 (steering + tau)
    nparam = model.p.rows()        # 1 (delayed command)
    nx = nx_augmented - 1          # 1 (only steering is measured)
    
    ny = R.shape[0] + Q.shape[0]                    # h(x), w
    ny_e = 0
    ny_0 = R.shape[0] + Q.shape[0] + Q0.shape[0]    # h(x), w and arrival cost
    
    # Set horizon
    ocp_mhe.solver_options.N_horizon = N
    
    x = ocp_mhe.model.x
    u = ocp_mhe.model.u  # process noise
    
    # Cost type
    ocp_mhe.cost.cost_type = 'NONLINEAR_LS'
    ocp_mhe.cost.cost_type_e = 'LINEAR_LS' 
    ocp_mhe.cost.cost_type_0 = 'NONLINEAR_LS'
    
    # Initial stage cost: [measurement, process_noise, arrival_cost]
    ocp_mhe.cost.W_0 = block_diag(R, Q, Q0)
    ocp_mhe.model.cost_y_expr_0 = vertcat(x[:nx], u, x)  # [steering_measured, w_steering, steering_state, tau_state]
    ocp_mhe.cost.yref_0 = np.zeros((ny_0,))
    
    # Intermediate stages: [measurement, process_noise]
    ocp_mhe.cost.W = block_diag(R, Q)
    ocp_mhe.model.cost_y_expr = vertcat(x[:nx], u)  # [steering_measured, w_steering]
    
    # Set parameter values
    ocp_mhe.parameter_values = np.zeros((nparam,))  # delayed commanded steering
    
    # Reference trajectories
    ocp_mhe.cost.yref = np.zeros((ny,))
    ocp_mhe.cost.yref_e = np.zeros((ny_e,))
    ocp_mhe.cost.yref_0 = np.zeros((ny_0,))
    
    # Bounds on states to ensure realistic values
    ocp_mhe.constraints.lbx = np.array([-np.pi, 0.05])  # steering, tau
    ocp_mhe.constraints.ubx = np.array([np.pi, 1.0])    # steering, tau (more realistic upper bound)
    ocp_mhe.constraints.idxbx = np.array([0, 1])
    
    # Bounds on process noise
    ocp_mhe.constraints.lbu = np.array([-0.1])  # w_steering
    ocp_mhe.constraints.ubu = np.array([0.1])   # w_steering
    ocp_mhe.constraints.idxbu = np.array([0])
    
    # Solver options (following pendulum example)
    ocp_mhe.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp_mhe.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp_mhe.solver_options.integrator_type = 'ERK'
    ocp_mhe.solver_options.tf = N * h
    ocp_mhe.solver_options.nlp_solver_type = 'SQP'
    ocp_mhe.solver_options.nlp_solver_max_iter = 200
    
    ocp_mhe.code_export_directory = 'steering_mhe_generated_code'
    
    # Create solver
    acados_solver_mhe = AcadosOcpSolver(ocp_mhe, json_file='steering_mhe.json')
    
    # Set arrival cost weighting
    acados_solver_mhe.cost_set(0, "W", block_diag(R, Q, Q0))
    
    return acados_solver_mhe

class SteeringMHE:
    def __init__(self, horizon=20, dt=0.1, delay=0.3, initial_tau=0.2):
        self.horizon = horizon
        self.dt = dt
        self.delay = delay  # Delay in seconds
        self.delay_samples = int(round(delay / dt))  # Convert to samples
        self.tau = initial_tau  # Use provided initial time constant estimate
        
        # Buffers for data with delay handling
        self.input_buffer = deque(maxlen=horizon + self.delay_samples + 10)
        self.measurement_buffer = deque(maxlen=horizon + 10)
        
        # Noise statistics (following pendulum example conventions)
        measurement_noise_std = 0.01   # measurement noise
        process_noise_std = 0.1       # process noise for steering
        arrival_cost_steering = 1.0   # arrival cost for steering
        arrival_cost_tau = 0.2      # high confidence in tau estimate
        
        # Cost matrices (correct dimensions)
        self.R = np.array([[1.0 / (measurement_noise_std**2)]])  # 1x1 for steering measurement
        self.Q = np.array([[1.0 / (process_noise_std**2)]])      # 1x1 for process noise
        self.Q0 = np.diag([arrival_cost_steering, arrival_cost_tau])  # 2x2 for arrival cost [steering, tau]
        
        # Setup MHE
        self.model = export_steering_mhe_model()
        self.solver = export_steering_mhe_solver(self.model, horizon, dt, self.Q, self.Q0, self.R)
        
        # Initialize solver
        self.x0_bar = np.array([0.0, self.tau])  # initial guess [steering, tau]
        for i in range(horizon + 1):
            self.solver.set(i, 'x', self.x0_bar)
            
        # Initialize with zero process noise
        for i in range(horizon):
            self.solver.set(i, 'u', np.array([0.0]))  # single process noise value
            
    def get_delayed_input(self, index):
        """Get delayed input for given index"""
        if len(self.input_buffer) > index + self.delay_samples:
            return self.input_buffer[-(index + self.delay_samples + 1)]
        else:
            return 0.0  # Default if not enough history
            
    def update(self, u_cmd, y_meas):
        """Update MHE with new commanded input and measurement"""
        
        # Add to buffers
        self.input_buffer.append(u_cmd)
        self.measurement_buffer.append(y_meas)
        
        # Need enough data for horizon and delay
        if len(self.measurement_buffer) < self.horizon or len(self.input_buffer) < self.horizon + self.delay_samples:
            return
            
        # Get recent data
        y_hist = list(self.measurement_buffer)[-self.horizon:]
        
        try:
            # Set initial stage (stage 0)
            yref_0 = np.zeros((4,))  # [measurement, w_steering, arrival_steering, arrival_tau]
            yref_0[0] = y_hist[0]     # measurement
            yref_0[1] = 0.0           # process noise reference (zero)
            yref_0[2:] = self.x0_bar  # arrival cost reference
            self.solver.set(0, "yref", yref_0)
            
            # Set delayed input for stage 0
            u_delayed_0 = self.get_delayed_input(self.horizon - 1)
            self.solver.set(0, "p", np.array([u_delayed_0]))
            
            # Set intermediate stages
            yref = np.zeros((2,))  # [measurement, w_steering] 
            for j in range(1, self.horizon):
                yref[0] = y_hist[j]  # measurement
                yref[1] = 0.0        # process noise reference (zero)
                self.solver.set(j, "yref", yref)
                
                # Set delayed input for this stage
                u_delayed_j = self.get_delayed_input(self.horizon - 1 - j)
                self.solver.set(j, "p", np.array([u_delayed_j]))
                
            # Solve MHE problem
            status = self.solver.solve()
            
            if status == 0:
                # Extract state estimate
                x_est = self.solver.get(self.horizon, "x")
                old_tau = self.tau
                self.tau = x_est[1]  # update time constant estimate
                
                # Update arrival cost mean for next iteration
                self.x0_bar = self.solver.get(1, "x")
                
                if abs(self.tau - old_tau) > 0.001:
                    print(f"MHE: tau updated from {old_tau:.6f} to {self.tau:.6f}")
                
            else:
                print(f"MHE solver failed with status: {status}")
                
        except Exception as e:
            print(f"Error in MHE update: {e}")
            import traceback
            traceback.print_exc()
            
    def get_time_constant(self):
        """Get current time constant estimate"""
        return self.tau
        
    def get_delay(self):
        """Get delay in seconds"""
        return self.delay
        
    def simulate_model(self, u_commands, initial_steering=0.0, simulator=None):
        """Simulate the estimated model for validation using acados ERK integration"""
        if simulator is not None:
            # Use provided simulator
            return simulator.simulate_trajectory(
                u_commands=u_commands,
                initial_steering=initial_steering,
                tau=self.tau,
                time_steps=None
            )
        else:
            # Fall back to convenience function
            return simulate_steering_system(
                u_commands=u_commands,
                tau=self.tau,
                delay_samples=self.delay_samples,
                initial_steering=initial_steering,
                dt=self.dt
            ) 