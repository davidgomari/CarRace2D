# agents/mpc_agent.py
from .base_agent import BaseAgent
import numpy as np
import casadi as ca

# Value to use for "far away" x/y coordinates for dummy opponents
FAR_AWAY_COORD = 1e6 

class MPCAgent(BaseAgent):
    def __init__(self, agent_id, mpc_params, car_params, track_params):
        super().__init__(agent_id)
        print(f"Initializing Modular MPC Agent {self.id}")

        # --- Store Parameters ---
        self.mpc_params = mpc_params
        self.car_params = car_params
        self.track_params = track_params

        # --- Core MPC Parameters ---
        self.N = self.mpc_params.get('horizon', 15) 
        self.dt = self.mpc_params.get('dt')
        if self.dt is None:
            raise ValueError("MPC Agent requires 'dt' in mpc_params.")

        # --- Car Physics Parameters ---
        self.L = self.car_params.get('wheelbase', 2.5)
        self.max_steer_vehicle = self.car_params.get('max_steer_angle', 0.6) # Max physical steer of car
        self.max_speed_vehicle = self.car_params.get('max_speed', 20.0)    # Max physical speed of car
        self.mass = self.car_params.get('mass', 1500.0) 
        self.max_engine_force = self.car_params.get('max_engine_force', 4500.0)
        self.max_brake_force = self.car_params.get('max_brake_force', 6000.0)
        self.ego_collision_radius = self.car_params.get('collision_radius', 1.5)

        # --- MPC-Specific Limits (can differ from physical limits for smoother control) ---
        self.mpc_max_accel_cmd = self.mpc_params.get('mpc_max_accel_cmd', self.car_params.get('max_accel', 3.0))
        self.mpc_min_accel_cmd = self.mpc_params.get('mpc_min_accel_cmd', self.car_params.get('min_accel', -5.0))
        self.mpc_max_steer_cmd = self.mpc_params.get('mpc_max_steer_cmd', self.max_steer_vehicle)
        self.mpc_max_steer_rate_cmd = self.mpc_params.get('mpc_max_steer_rate_cmd', np.pi) # Max rate of change for target_steer_cmd

        # --- Track Parameters ---
        self.track_width = self.track_params.get('width', 10.0)

        # --- Multi-Agent Collision Avoidance Parameters ---
        self.max_opponents_for_constraint = self.mpc_params.get('max_opponents_to_consider', 0)
        self.default_opponent_radius = self.mpc_params.get('default_opponent_collision_radius', 1.5)

        # --- Configuration Flags for Enabling/Disabling Features ---
        # Cost components
        self.cfg_use_progress_cost = self.mpc_params.get('use_progress_cost', True)
        self.cfg_use_path_tracking_cost = self.mpc_params.get('use_path_tracking_cost', True)
        self.cfg_use_heading_cost = self.mpc_params.get('use_heading_cost', True)
        self.cfg_use_velocity_cost = self.mpc_params.get('use_velocity_cost', True)
        self.cfg_use_control_effort_cost = self.mpc_params.get('use_control_effort_cost', True)
        self.cfg_use_steer_rate_cost = self.mpc_params.get('use_steer_rate_cost', True)
        self.cfg_use_terminal_cost = self.mpc_params.get('use_terminal_cost', True)
        # Constraints
        self.cfg_apply_track_boundaries = self.mpc_params.get('apply_track_boundaries', True)
        self.cfg_apply_collision_avoidance = self.mpc_params.get('apply_collision_avoidance', self.max_opponents_for_constraint > 0)
        self.cfg_use_soft_track_boundaries = self.mpc_params.get('use_soft_track_boundaries', False) # Example for soft constraints
        self.cfg_soft_boundary_penalty = self.mpc_params.get('soft_boundary_penalty', 1000.0)

        # --- CasADi Optimization Setup ---
        self.opti = ca.Opti()
        self._define_decision_variables()
        self._define_cost_function()
        self._define_dynamics_constraints()
        self._define_state_and_control_constraints()
        
        if self.cfg_apply_track_boundaries:
            self._define_track_boundary_constraints()
        
        if self.cfg_apply_collision_avoidance and self.max_opponents_for_constraint > 0:
            self._define_collision_avoidance_constraints()

        # Initial condition constraint (always applied)
        self.opti.subject_to(self.X[:, 0] == self.X0_param)

        # --- Solver Setup ---
        solver_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        # User-configurable solver options
        user_solver_opts = self.mpc_params.get('solver_options', {})
        solver_opts.update(user_solver_opts) # Override defaults with user settings
        
        self.opti.solver('ipopt', solver_opts)
        self.warm_start_enabled = False 
        self.sol_X_prev = None
        self.sol_U_prev = None
        print(f"Modular MPC Agent {self.id} Initialized. Collision Avoidance: {self.cfg_apply_collision_avoidance}, Track Boundaries: {self.cfg_apply_track_boundaries}")

    def _define_decision_variables(self):
        """Defines state, control, and parameter variables for the OCP."""
        # State variables: [x, y, v, theta, steer_angle_vehicle] (5 states)
        self.X = self.opti.variable(5, self.N + 1)
        self.x_pos_var = self.X[0, :]
        self.y_pos_var = self.X[1, :]
        self.v_var = self.X[2, :]
        self.theta_var = self.X[3, :]
        self.steer_angle_vehicle_var = self.X[4, :] # Actual steering angle of the vehicle

        # Control variables: [accel_cmd, target_steer_angle_cmd] (2 controls)
        self.U = self.opti.variable(2, self.N)
        self.a_cmd_var = self.U[0, :]            # Desired longitudinal acceleration
        self.target_steer_cmd_var = self.U[1, :] # Desired target steering angle for the wheels

        # Parameters
        self.X0_param = self.opti.parameter(5) # Initial state
        self.P_ref_param = self.opti.parameter(4, self.N) # Reference path [x_ref, y_ref, theta_ref, v_ref]

        # Parameters for opponent predictions (if collision avoidance is enabled)
        if self.cfg_apply_collision_avoidance and self.max_opponents_for_constraint > 0:
            self.opponent_pred_x_params = [self.opti.parameter(self.N + 1) for _ in range(self.max_opponents_for_constraint)]
            self.opponent_pred_y_params = [self.opti.parameter(self.N + 1) for _ in range(self.max_opponents_for_constraint)]
            self.opponent_radii_params = [self.opti.parameter() for _ in range(self.max_opponents_for_constraint)]
        else:
            self.opponent_pred_x_params = []
            self.opponent_pred_y_params = []
            self.opponent_radii_params = []
        
        # Slack variables for soft constraints (if used)
        if self.cfg_apply_track_boundaries and self.cfg_use_soft_track_boundaries:
            self.slack_track_boundary = self.opti.variable(self.N + 1)
            self.opti.subject_to(self.slack_track_boundary >= 0) # Slack must be non-negative

        # Add CasADi parameter for reference progress (for progress-only cost)
        self.reference_progress_param = self.opti.parameter()

    def _define_cost_function(self):
        """Defines the cost function for the OCP based on configuration."""
        cost = 0

        # --- Cost Weights (from mpc_params) ---
        Q_pos = self.mpc_params.get('Q_pos', 10.0)        
        Q_head = self.mpc_params.get('Q_head', 2.0)       
        Q_vel = self.mpc_params.get('Q_vel', 1.0)         
        R_accel = self.mpc_params.get('R_accel', 0.1)         
        R_steer_cmd = self.mpc_params.get('R_steer_cmd', 0.1) 
        R_steer_rate = self.mpc_params.get('R_steer_rate', 0.05)
        
        # Terminal cost weights (can be different)
        Q_pos_term = self.mpc_params.get('Q_pos_terminal', Q_pos)        
        Q_head_term = self.mpc_params.get('Q_head_terminal', Q_head)       
        Q_vel_term = self.mpc_params.get('Q_vel_terminal', Q_vel)   

        # --- Stage Costs (Loop over prediction horizon N) ---
        Q_progress = self.mpc_params.get('Q_progress', 1.0)

        if self.cfg_use_path_tracking_cost:
            cost += Q_pos * ca.sumsqr(self.x_pos_var[:self.N] - self.P_ref_param[0, :]) \
                 + Q_pos * ca.sumsqr(self.y_pos_var[:self.N] - self.P_ref_param[1, :])

        if self.cfg_use_heading_cost:
            heading_error = self.theta_var[:self.N] - self.P_ref_param[2, :]
            cost += Q_head * ca.sumsqr(1 - ca.cos(heading_error))



        for k in range(self.N):
            # if self.cfg_use_path_tracking_cost:
            #     cost += Q_pos * ((self.x_pos_var[k] - self.P_ref_param[0, k])**2 + \
            #                      (self.y_pos_var[k] - self.P_ref_param[1, k])**2)
            
            # if self.cfg_use_heading_cost:
            #     heading_error = self.theta_var[k] - self.P_ref_param[2, k]
            #     cost += Q_head * (1 - ca.cos(heading_error)) # Robust to angle wrapping

            if self.cfg_use_velocity_cost:
                # print(f"v_ref:{self.P_ref_param[3, k]}\t v:{self.v_var[k]}")
                cost += Q_vel * (self.v_var[k] - self.P_ref_param[3, k])**2
            
            if self.cfg_use_control_effort_cost:
                cost += R_accel * self.a_cmd_var[k]**2
                cost += R_steer_cmd * self.target_steer_cmd_var[k]**2

            if self.cfg_use_steer_rate_cost:
                if k > 0:
                    cost += R_steer_rate * (self.target_steer_cmd_var[k] - self.target_steer_cmd_var[k-1])**2
                else: # Penalize first steering command relative to current actual vehicle steer
                    cost += R_steer_rate * (self.target_steer_cmd_var[k] - self.X0_param[4])**2
        
        # --- Terminal Cost (at step N) ---
        if self.cfg_use_terminal_cost:
            k_terminal = self.N # State at X[:, N]
            ref_idx_terminal = self.N -1 # Reference is for N steps (0 to N-1)

            cost += Q_pos_term * ((self.x_pos_var[k_terminal] - self.P_ref_param[0, ref_idx_terminal])**2 + \
                                  (self.y_pos_var[k_terminal] - self.P_ref_param[1, ref_idx_terminal])**2)
            
            if self.cfg_use_path_tracking_cost: # Apply terminal path tracking
                cost += Q_pos_term * ((self.x_pos_var[k_terminal] - self.P_ref_param[0, ref_idx_terminal])**2 + \
                                      (self.y_pos_var[k_terminal] - self.P_ref_param[1, ref_idx_terminal])**2)
            if self.cfg_use_heading_cost: # Apply terminal heading tracking
                terminal_heading_error = self.theta_var[k_terminal] - self.P_ref_param[2, ref_idx_terminal]
                cost += Q_head_term * (1 - ca.cos(terminal_heading_error))
            if self.cfg_use_velocity_cost: # Apply terminal velocity tracking
                cost += Q_vel_term * (self.v_var[k_terminal] - self.P_ref_param[3, ref_idx_terminal])**2
        
        # Add penalty for soft track boundary slack variables
        if self.cfg_apply_track_boundaries and self.cfg_use_soft_track_boundaries:
            cost += self.cfg_soft_boundary_penalty * ca.sumsqr(self.slack_track_boundary)

        # Wee need to define Progress Cost HERE (all other costs are disabled)

        if self.cfg_use_progress_cost:
            cost += -Q_progress * self.reference_progress_param

        self.opti.minimize(cost)

    def _define_dynamics_constraints(self):
        """Defines the vehicle dynamics constraints."""
        for k in range(self.N):
            current_states = self.X[:, k]
            current_controls = self.U[:, k]
            state_derivatives = self._vehicle_dynamics_casadi(current_states, current_controls)
            x_next_rk4 = current_states + self.dt * state_derivatives # Euler integration
            # # Example for RK4 (more complex to implement correctly with CasADi)
            # k1 = self.dt * self._vehicle_dynamics_casadi(current_states, current_controls)
            # k2 = self.dt * self._vehicle_dynamics_casadi(current_states + 0.5 * k1, current_controls)
            # k3 = self.dt * self._vehicle_dynamics_casadi(current_states + 0.5 * k2, current_controls)
            # k4 = self.dt * self._vehicle_dynamics_casadi(current_states + k3, current_controls)
            # x_next_rk4 = current_states + (k1 + 2*k2 + 2*k3 + k4) / 6.0
            self.opti.subject_to(self.X[:, k+1] == x_next_rk4)

    def _vehicle_dynamics_casadi(self, state_sx, control_sx):
        """ CasADi symbolic vehicle dynamics. """
        # state_sx: [x, y, v, theta, steer_angle_vehicle]
        # control_sx: [accel_cmd, target_steer_cmd]
        v = state_sx[2]
        theta = state_sx[3]
        steer_angle_vehicle = state_sx[4] # Current actual steer angle of the vehicle
        
        accel_cmd = control_sx[0]          # Commanded longitudinal acceleration
        target_steer_cmd = control_sx[1]   # Commanded target steering angle

        x_dot = v * ca.cos(theta)
        y_dot = v * ca.sin(theta)
        v_dot = accel_cmd 
        theta_dot = (v / self.L) * ca.tan(steer_angle_vehicle) 
        
        # Model steering actuator dynamics: actual steer angle moves towards target steer command
        steer_error = target_steer_cmd - steer_angle_vehicle
        # Limit the rate of change of the actual steering angle
        steer_angle_vehicle_dot = ca.fmax(-self.mpc_max_steer_rate_cmd, 
                                          ca.fmin(self.mpc_max_steer_rate_cmd, steer_error / self.dt))
        # The above implies that if target_steer_cmd is reachable within dt at max_steer_rate, it will reach it.
        # Otherwise, it changes by max_steer_rate_cmd * dt.

        return ca.vertcat(x_dot, y_dot, v_dot, theta_dot, steer_angle_vehicle_dot)

    def _define_state_and_control_constraints(self):
        """Defines basic state and control input limits."""
        # Control command limits
        self.opti.subject_to(self.opti.bounded(self.mpc_min_accel_cmd, self.a_cmd_var, self.mpc_max_accel_cmd))
        self.opti.subject_to(self.opti.bounded(-self.mpc_max_steer_cmd, self.target_steer_cmd_var, self.mpc_max_steer_cmd))

        # Vehicle state limits (applied to the predicted states)
        self.opti.subject_to(self.opti.bounded(0, self.v_var, self.max_speed_vehicle)) # Velocity non-negative and capped
        self.opti.subject_to(self.opti.bounded(-self.max_steer_vehicle, self.steer_angle_vehicle_var, self.max_steer_vehicle)) # Actual vehicle steer angle

    def _define_track_boundary_constraints(self):
        """Defines track boundary constraints."""
        for k in range(self.N + 1): 
            dx_k = self.x_pos_var[k] - self.P_ref_param[0, min(k, self.N-1)] 
            dy_k = self.y_pos_var[k] - self.P_ref_param[1, min(k, self.N-1)]
            ref_theta_k = self.P_ref_param[2, min(k, self.N-1)]
            
            # Lateral deviation from the reference path
            # Positive if to the left of reference path, negative if to the right.
            predicted_lateral_deviation = dx_k * (-ca.sin(ref_theta_k)) + dy_k * (ca.cos(ref_theta_k))
            
            half_track_width = self.track_width / 2.0
            if self.cfg_use_soft_track_boundaries:
                self.opti.subject_to(predicted_lateral_deviation <= half_track_width + self.slack_track_boundary[k])
                self.opti.subject_to(predicted_lateral_deviation >= -half_track_width - self.slack_track_boundary[k])
            else:
                self.opti.subject_to(self.opti.bounded(-half_track_width,
                                                       predicted_lateral_deviation,
                                                       half_track_width))

    def _define_collision_avoidance_constraints(self):
        """Defines collision avoidance constraints with other agents."""
        for i in range(self.max_opponents_for_constraint):
            opponent_x_traj_param = self.opponent_pred_x_params[i]
            opponent_y_traj_param = self.opponent_pred_y_params[i]
            opponent_radius_param = self.opponent_radii_params[i]

            for k in range(self.N + 1): 
                dx_opponent = self.x_pos_var[k] - opponent_x_traj_param[k]
                dy_opponent = self.y_pos_var[k] - opponent_y_traj_param[k]
                
                sum_radii = self.ego_collision_radius + opponent_radius_param
                # Constraint: actual_dist_sq >= sum_radii_sq
                self.opti.subject_to(dx_opponent**2 + dy_opponent**2 >= sum_radii**2)

    def get_action(self, observation):
        required_keys = ['x', 'y', 'v', 'theta', 'steer_angle', 'reference_path']
        for key in required_keys:
            if key not in observation:
                # print(f"[MPC DEBUG] Missing key '{key}' in observation for agent {self.id}. Observation keys: {list(observation.keys())}")
                return np.array([0.0, 0.0], dtype=np.float32)

        # Check and print the type and shape of each required observation
        for key in required_keys:
            val = observation[key]
            # print(f"[MPC DEBUG] Observation[{key}] type: {type(val)}, value: {val if isinstance(val, (float, int)) else (val.shape if hasattr(val, 'shape') else val)}")

        # Extract and check values
        try:
            current_x = float(observation['x'][0]) if isinstance(observation['x'], (list, np.ndarray)) else float(observation['x'])
            current_y = float(observation['y'][0]) if isinstance(observation['y'], (list, np.ndarray)) else float(observation['y'])
            current_v = float(observation['v'][0]) if isinstance(observation['v'], (list, np.ndarray)) else float(observation['v'])
            current_theta = float(observation['theta'][0]) if isinstance(observation['theta'], (list, np.ndarray)) else float(observation['theta'])
            current_steer = float(observation['steer_angle'][0]) if isinstance(observation['steer_angle'], (list, np.ndarray)) else float(observation['steer_angle'])
        except Exception as e:
            print(f"[MPC DEBUG] Error extracting state values: {e}")
            print(f"[MPC DEBUG] Raw observation: {observation}")
            return np.array([0.0, 0.0], dtype=np.float32)

        current_state_vec = np.array([current_x, current_y, current_v, current_theta, current_steer])
        # print(f"[MPC DEBUG] current_state_vec: {current_state_vec}")
        self.opti.set_value(self.X0_param, current_state_vec)

        ref_path_raw = np.array(observation['reference_path'])
        # print(f"[MPC DEBUG] reference_path raw, shape: {ref_path_raw.shape}, values (first 5): \n{ref_path_raw[:5] if ref_path_raw.shape[0] > 4 else ref_path_raw}")
        if ref_path_raw.shape[0] == self.N and ref_path_raw.shape[1] == 4: # Expected (N, 4)
            ref_path_for_casadi = ref_path_raw.T # Transpose to (4, N) for CasADi
        elif ref_path_raw.shape[0] == 4 and ref_path_raw.shape[1] == self.N: # Already (4, N)
            ref_path_for_casadi = ref_path_raw
        else:
            # print(f"[MPC DEBUG] reference_path shape {ref_path_raw.shape} not ({self.N}, 4) or (4, {self.N}).")
            # print(f"[MPC DEBUG] reference_path content: {ref_path_raw}")
            return np.array([-0.1, current_steer], dtype=np.float32) # Brake gently
        self.opti.set_value(self.P_ref_param, ref_path_for_casadi)

        # Check for out-of-bounds reference velocities
        v_refs = ref_path_for_casadi[3, :]
        # if np.any(v_refs > self.max_speed_vehicle * 1.1) or np.any(v_refs < 0):
            # print(f"[MPC DEBUG] Reference velocities out of bounds: {v_refs}")
            # print(f"[MPC DEBUG] max_speed_vehicle: {self.max_speed_vehicle}")

        # Set opponent trajectory parameters if collision avoidance is active
        if self.cfg_apply_collision_avoidance and self.max_opponents_for_constraint > 0:
            opponent_predictions = observation.get('opponent_predicted_trajectories', [])
            num_actual_opponents = len(opponent_predictions)
            # print(f"[MPC DEBUG] Collision avoidance enabled. num_actual_opponents: {num_actual_opponents}")
            for i in range(self.max_opponents_for_constraint):
                if i < num_actual_opponents:
                    pred = opponent_predictions[i]
                    if 'x_traj' in pred and 'y_traj' in pred and \
                       len(pred['x_traj']) == self.N + 1 and \
                       len(pred['y_traj']) == self.N + 1:
                        self.opti.set_value(self.opponent_pred_x_params[i], pred['x_traj'])
                        self.opti.set_value(self.opponent_pred_y_params[i], pred['y_traj'])
                        self.opti.set_value(self.opponent_radii_params[i], pred.get('radius', self.default_opponent_radius))
                    else:
                        # print(f"[MPC DEBUG] Opponent {i} prediction data malformed or length mismatch. Using dummy.")
                        self._set_dummy_opponent_params(i)
                else:
                    # print(f"[MPC DEBUG] No opponent for slot {i}, using dummy.")
                    self._set_dummy_opponent_params(i)

        # Set reference progress parameter if present
        if self.cfg_use_progress_cost and 'reference_progress' in observation:
            # print(float(observation['reference_progress']))
            self.opti.set_value(self.reference_progress_param, float(observation['reference_progress']))

        # Warm start
        if self.warm_start_enabled and self.sol_X_prev is not None and self.sol_U_prev is not None:
            try:
                if self.sol_X_prev.shape == (5, self.N + 1) and self.sol_U_prev.shape == (2, self.N):
                    self.opti.set_initial(self.X, self.sol_X_prev)
                    self.opti.set_initial(self.U, self.sol_U_prev)
                    if self.cfg_apply_track_boundaries and self.cfg_use_soft_track_boundaries and self.sol_slack_prev is not None:
                         if self.sol_slack_prev.shape == (self.N + 1,): # Check shape
                            self.opti.set_initial(self.slack_track_boundary, self.sol_slack_prev)
                else:
                    # print(f"[MPC DEBUG] Warm start shapes do not match expected. Disabling warm start.")
                    self.warm_start_enabled = False 
            except Exception as e:
                print(f"[MPC DEBUG] Exception during warm start: {e}")
                self.warm_start_enabled = False 
        else:
             self.warm_start_enabled = False

        try:
            sol = self.opti.solve()
            optimal_U = sol.value(self.U)
            mpc_accel_cmd, mpc_target_steer_cmd = optimal_U[:, 0] 

            self.sol_X_prev = sol.value(self.X) 
            self.sol_U_prev = sol.value(self.U)
            if self.cfg_apply_track_boundaries and self.cfg_use_soft_track_boundaries:
                self.sol_slack_prev = sol.value(self.slack_track_boundary)
            self.warm_start_enabled = True

            # Convert accel_cmd to throttle/brake input for the car environment
            if mpc_accel_cmd >= 0:
                throttle_brake_input = mpc_accel_cmd / self.mpc_max_accel_cmd if self.mpc_max_accel_cmd > 1e-6 else 0
            else:
                throttle_brake_input = mpc_accel_cmd / abs(self.mpc_min_accel_cmd) if abs(self.mpc_min_accel_cmd) > 1e-6 else 0
            throttle_brake_input = np.clip(throttle_brake_input, -1.0, 1.0)
            final_steer_cmd = np.clip(mpc_target_steer_cmd, -self.max_steer_vehicle, self.max_steer_vehicle)
            action = np.array([throttle_brake_input, final_steer_cmd])
            # print(f"[MPC DEBUG] Action output: {action}")
            return action.astype(np.float32)

        except Exception as e:
            print(f"[MPC DEBUG] Solver failed: {e}")
            print(f"[MPC DEBUG] Initial State for MPC: {current_state_vec}")
            print(f"[MPC DEBUG] Reference Path (for CASADI) (first 5): \n{ref_path_for_casadi[:5,:] if ref_path_for_casadi.shape[0] > 4 else ref_path_for_casadi}")
            if self.cfg_apply_collision_avoidance and self.max_opponents_for_constraint > 0:
                print(f"[MPC DEBUG] Opponent predictions: {observation.get('opponent_predicted_trajectories', [])}")
            self.warm_start_enabled = False
            self.sol_X_prev = None 
            self.sol_U_prev = None
            if self.cfg_apply_track_boundaries and self.cfg_use_soft_track_boundaries:
                self.sol_slack_prev = None
            return np.array([-0.2, current_steer], dtype=np.float32) # Brake gently

    def _set_dummy_opponent_params(self, opponent_idx):
        """Sets CasADi parameters for a dummy/non-existent opponent to be far away."""
        if not (self.cfg_apply_collision_avoidance and self.max_opponents_for_constraint > 0):
            return # Do nothing if collision avoidance is off or no slots

        dummy_x_traj = np.full(self.N + 1, FAR_AWAY_COORD)
        dummy_y_traj = np.full(self.N + 1, FAR_AWAY_COORD)
        dummy_radius = 0.01 # Small radius, effectively making it a point far away

        self.opti.set_value(self.opponent_pred_x_params[opponent_idx], dummy_x_traj)
        self.opti.set_value(self.opponent_pred_y_params[opponent_idx], dummy_y_traj)
        self.opti.set_value(self.opponent_radii_params[opponent_idx], dummy_radius)

