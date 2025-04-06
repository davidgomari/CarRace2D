# # agents/mpc_agent.py (Placeholder)
# from .base_agent import BaseAgent
# import numpy as np

# class MPCAgent(BaseAgent):
#     def __init__(self, agent_id, params):
#         super().__init__(agent_id)
#         self.params = params
#         # TODO: Initialize the MPC controller (e.g., using CasADi, acados)
#         print(f"MPC Agent {self.id}: Initialization placeholder.")

#     def get_action(self, observation):
#         # TODO: Implement MPC logic
#         # 1. Get current state from observation
#         # 2. Define prediction horizon and cost function
#         # 3. Solve the optimal control problem
#         # 4. Return the first action from the optimal sequence
#         print(f"MPC Agent {self.id}: Returning zero action (placeholder).")
#         return np.array([0.0, 0.0], dtype=np.float32)


# agents/mpc_agent.py
from .base_agent import BaseAgent
import numpy as np
import casadi as ca

class MPCAgent(BaseAgent):
    def __init__(self, agent_id, mpc_params, car_params, track_params):
        super().__init__(agent_id)
        print(f"Initializing MPC Agent {self.id}")

        # MPC Parameters
        self.N = mpc_params.get('horizon', 10)
        self.dt = mpc_params.get('dt')
        if self.dt is None:
            raise ValueError("MPC Agent requires 'dt' in mpc_params (should be injected by environment).")

        # Car Parameters (Extract ALL needed for MPC logic and action conversion)
        self.L = car_params.get('wheelbase', 2.5)
        self.max_steer = car_params.get('max_steer_angle', 0.6)
        self.max_speed = car_params.get('max_speed', 20.0)
        # Store parameters needed for action conversion
        self.mass = car_params.get('mass', 1500.0)
        self.max_engine_force = car_params.get('max_engine_force', 4500.0)
        self.max_brake_force = car_params.get('max_brake_force', 6000.0)
        # Store parameters needed for MPC constraints/cost (if different from action limits)
        # Note: MPC internal limits might differ from overall car physics limits
        self.mpc_max_accel = car_params.get('max_accel', 3.0) # Example: MPC might use simpler accel limits
        self.mpc_min_accel = car_params.get('min_accel', -5.0) # Example: MPC might use simpler accel limits 

        # Track Parameters (Extract needed)
        self.track_width = track_params.get('width', 10.0)
        self.track_radius = track_params.get('radius', 30.0)
        self.track_length = track_params.get('length', 100.0)

        # --- CasADi Optimization Setup ---
        self.opti = ca.Opti()

        # State variables [x, y, v, theta]
        self.X = self.opti.variable(4, self.N + 1)
        x_var = self.X[0, :]
        y_var = self.X[1, :]
        v_var = self.X[2, :]
        theta_var = self.X[3, :]

        # Control variables [accel, steer_angle]
        self.U = self.opti.variable(2, self.N)
        a_var = self.U[0, :]
        delta_var = self.U[1, :]

        # Initial state parameter (will be set at each step)
        self.X0 = self.opti.parameter(4)

        # --- Cost Function ---
        cost = 0
        # Reference tracking, speed maximization, control minimization
        Q_speed = -1.0
        Q_track = 10.0
        R_accel = 0.5
        R_steer = 1.0
        # Add penalty for deviation from track centerline (more robust needed)
        # Q_centerline = 5.0 

        for k in range(self.N):
            cost += Q_speed * v_var[k]
            # TODO: Refine track cost - use distance to actual centerline if available
            cost += Q_track * y_var[k]**2 # Penalize y deviation (simple)
            # Example using centerline distance if available in obs (not standard MPC state):
            # If 'dist_to_centerline' were added as a *parameter* based on obs:
            # dist_param = self.opti.parameter() 
            # cost += Q_centerline * dist_param**2 
            cost += R_accel * a_var[k]**2
            cost += R_steer * delta_var[k]**2

        self.opti.minimize(cost)

        # --- Dynamics Constraints ---
        for k in range(self.N):
            x_next = self.X[:, k] + self.dt * self.dynamics_func(self.X[:, k], self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == x_next)

        # --- Boundary Constraints (Using stored MPC-specific accel limits) ---
        self.opti.subject_to(self.opti.bounded(self.mpc_min_accel, a_var, self.mpc_max_accel))
        self.opti.subject_to(self.opti.bounded(-self.max_steer, delta_var, self.max_steer))
        self.opti.subject_to(self.opti.bounded(0, v_var, self.max_speed))
        # self.opti.subject_to(y_var <= self.track_width / 2)
        # self.opti.subject_to(y_var >= -self.track_width / 2)

        self.opti.subject_to(self.X[:, 0] == self.X0)

        # --- Solver Setup ---
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)
        self.warm_start_enabled = False # Flag to track if warm start values exist
        print(f"MPC Agent {self.id} Initialized.")

    def dynamics_func(self, state, control):
        """ Kinematic bicycle model dynamics for CasADi. """
        x, y, v, theta = state[0], state[1], state[2], state[3]
        a, delta = control[0], control[1]
        tan_delta = ca.tan(delta)
        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        v_dot = a
        theta_dot = ca.if_else(ca.fabs(v) < 1e-4, 0.0, (v / self.L) * tan_delta)
        return ca.vertcat(x_dot, y_dot, v_dot, theta_dot)

    def get_action(self, observation):
        """ Gets action based on the observation dictionary. """
        # Extract the core state needed for MPC from the dictionary
        required_keys = ['x', 'y', 'v', 'theta']
        if not all(key in observation for key in required_keys):
            print(f"MPC Agent {self.id} Error: Missing required keys in observation: {observation.keys()}")
            # Return safe action if observation is incomplete
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Extract values, ensuring they are scalars for set_value
        current_state = np.array([observation[key][0] for key in required_keys])
        # print(f"MPC Current State: {current_state}") # Debug

        # Set the initial state parameter
        self.opti.set_value(self.X0, current_state)

        # Optional: Set other parameters based on observation (e.g., dist_to_centerline if used in cost)
        # if 'dist_to_centerline' in observation:
        #    self.opti.set_value(self.dist_param, observation['dist_to_centerline'][0])

        # Provide initial guess (warm start) - Improves solver speed
        if self.warm_start_enabled:
            self.opti.set_initial(self.X, self.sol_X_prev)
            self.opti.set_initial(self.U, self.sol_U_prev)

        try:
            # Solve the optimization problem
            sol = self.opti.solve()

            # Extract the first optimal control action [accel, steer_angle]
            optimal_U = sol.value(self.U)
            mpc_accel, mpc_steer = optimal_U[:, 0]

            # Store solution for warm start next time
            self.sol_X_prev = sol.value(self.X)
            self.sol_U_prev = sol.value(self.U)
            self.warm_start_enabled = True
            
            # --- Convert MPC acceleration output to throttle/brake input [-1, 1] --- 
            if mpc_accel >= 0: # Acceleration -> Throttle
                # Scale based on max engine force (assumes linear relation for simplicity)
                throttle_brake_input = mpc_accel * self.mass / self.max_engine_force if self.max_engine_force > 0 else 0
            else: # Deceleration -> Brake
                 # Scale based on max brake force (input is negative)
                throttle_brake_input = mpc_accel * self.mass / self.max_brake_force if self.max_brake_force > 0 else 0
            
            # Clip to [-1, 1] range
            throttle_brake_input = np.clip(throttle_brake_input, -1.0, 1.0)
            # Clip steering angle (redundant as MPC should respect bounds, but safe)
            final_steer = np.clip(mpc_steer, -self.max_steer, self.max_steer)
            
            # Assemble final action: [throttle_brake_input, steer_angle]
            action = np.array([throttle_brake_input, final_steer])

            return action.astype(np.float32)

        except Exception as e:
            print(f"MPC Agent {self.id} Solver failed: {e}")
            print(f"Initial State: {current_state}")
            # Solver failure, reset warm start and return safe action
            self.warm_start_enabled = False 
            return np.array([0.0, 0.0], dtype=np.float32)


# --- Modifications needed in environment.py to initialize MPC agent ---
# Inside RacingEnv.__init__ where agents are created:

# elif agent_type == 'mpc':
#     mpc_params = agent_info.get('mpc_params', {}) # Get MPC specific params from config if provided
#     mpc_params['dt'] = self.dt # Pass environment dt to MPC
#     self.agents[agent_id] = MPCAgent(agent_id, mpc_params, car_cfg, track_cfg)

# --- Add mpc_params to config.yaml if needed ---
# agents:
#   agent_2:
#     type: 'mpc'
#     start_pos_idx: 2
#     mpc_params:
#       horizon: 15
#       # dt will be set from simulation dt