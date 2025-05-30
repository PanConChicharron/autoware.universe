import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat

def export_steering_model():
    """Export first-order steering system model for both MHE and simulation"""
    
    model_name = 'first_order_steering_model'
    
    # States: [steering_angle, time_constant]
    steering = SX.sym('steering')
    tau = SX.sym('tau')  # time constant as augmented state
    x = vertcat(steering, tau)
    
    # Control input (delayed commanded steering)
    u_delayed = SX.sym('u_delayed')
    u = u_delayed
    
    # State derivatives
    steering_dot = SX.sym('steering_dot')
    tau_dot = SX.sym('tau_dot')
    xdot = vertcat(steering_dot, tau_dot)
    
    # First-order dynamics with delay already incorporated in the input
    # d(steering)/dt = (-steering + u_delayed) / tau
    # d(tau)/dt = 0 (parameter evolves very slowly, almost constant)
    f_expl = vertcat(
        (-steering + u_delayed) / tau,
        0  # tau is nearly constant (parameter)
    )
    
    f_impl = xdot - f_expl
    
    model = AcadosModel()
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    
    return model

def export_steering_mhe_model():
    """Export first-order steering system MHE model with process noise"""
    
    model_name = 'first_order_steering_mhe_model'
    
    # States: [steering_angle, time_constant]
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