# car.py
import numpy as np

# OK

class Car:
    def __init__(self, car_id, initial_state, params, color='blue'):
        self.id = car_id
        self.params = params # Store the whole dict
        self.color = color # For visualization
        
        # Extract physics parameters (provide defaults if missing for backward compatibility)
        self.wheelbase = params.get('wheelbase', 2.5)
        self.mass = params.get('mass', 1500.0)
        self.inv_mass = 1.0 / self.mass
        self.max_v = params.get('max_speed', 20.0)
        self.max_steer = params.get('max_steer_angle', 0.6)
        self.collision_radius = params.get('collision_radius', 1.5)
        
        self.coeff_drag = params.get('coeff_drag', 0.8)
        self.coeff_rolling = params.get('coeff_rolling_resistance', 60.0)
        self.coeff_friction = params.get('coeff_friction', 1.1)
        self.coeff_cornering = params.get('coeff_cornering_stiffness', 15.0)
        self.max_engine_force = params.get('max_engine_force', 4500.0)
        self.max_brake_force = params.get('max_brake_force', 6000.0)
        self.gravity = params.get('gravity', 9.81)

        # Fallback if new force params aren't in config (old behavior)
        # Note: These are now less meaningful if forces are defined
        self.min_accel_legacy = params.get('min_accel', -5.0)
        self.max_accel_legacy = params.get('max_accel', 3.0)
        self.use_legacy_accel = not ('max_engine_force' in params and 'max_brake_force' in params)
        if self.use_legacy_accel:
            print(f"Car {self.id}: Warning - Using legacy min/max_accel. Define forces in config for new physics.")

        # State variables initialized in set_state
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.theta = 0.0
        self.steer_angle = 0.0 
        self.accel_lon = 0.0 # Store actual longitudinal acceleration applied
        self.accel_lat = 0.0 # Store actual lateral acceleration applied
        
        self.lap_count = 0

        # Progress tracking
        self.total_progress = 0.0
        self.last_x = 0.0
        self.last_y = 0.0
        
        self.set_state(initial_state) # Initialize state variables properly

    def set_state(self, state):
        self.x, self.y, self.v, self.theta = state
        self.steer_angle = 0.0 # Reset steer on state set
        self.accel_lon = 0.0 # Reset accel on state set
        self.accel_lat = 0.0 # Reset accel on state set
        self.last_x = self.x # Ensure last_x is updated
        self.last_y = self.y # Ensure last_y is updated
        self.total_progress = 0.0 # Reset total progress

    def update(self, action, dt):
        throttle_brake_input, steer_input = action # Action is now [-1, 1] for throttle/brake

        # --- Steering --- 
        # Apply steering constraints (simple direct mapping for now)
        self.steer_angle = np.clip(steer_input, -self.max_steer, self.max_steer)

        # Store previous x for lap counting
        self.last_x = self.x

        # --- Forces --- 
        # 1. Engine/Brake Force (based on input)
        if self.use_legacy_accel:
            # Fallback to old direct acceleration input
            force_drive_brake = np.clip(throttle_brake_input, self.min_accel_legacy, self.max_accel_legacy) * self.mass
        else:
            if throttle_brake_input > 0: # Acceleration
                force_drive_brake = throttle_brake_input * self.max_engine_force
            else: # Braking
                force_drive_brake = throttle_brake_input * self.max_brake_force # Input is negative

        # 2. Drag Force (depends on v^2)
        force_drag = -self.coeff_drag * self.v**2 * np.sign(self.v)

        # 3. Rolling Resistance (proportional to v, simplified)
        force_rolling = -self.coeff_rolling * self.v 

        # 4. Net Longitudinal Force
        force_longitudinal = force_drive_brake + force_drag + force_rolling
        
        # --- Acceleration Calculation --- 
        # Desired Longitudinal Acceleration
        accel_lon_desired = force_longitudinal * self.inv_mass 

        # Desired Lateral Acceleration (from Kinematic Bicycle Model approximation)
        # theta_dot = (v / L) * tan(steer) => approximates centripetal accel a_lat = v * theta_dot = (v^2 / L) * tan(steer)
        if abs(self.v) < 0.1: # Avoid instability at low speed
            accel_lat_desired = 0.0
        else:
             accel_lat_desired = (self.v**2 / self.wheelbase) * np.tan(self.steer_angle) 
             # Limit lateral accel based on simplified tire stiffness (prevents extreme values at high speed/steer)
             # This is a major simplification - real tires are much more complex!
             max_lat_accel_grip = self.coeff_cornering * self.gravity # Simplified grip limit
             accel_lat_desired = np.clip(accel_lat_desired, -max_lat_accel_grip, max_lat_accel_grip)

        # --- Friction Circle Constraint --- 
        # Calculate maximum total acceleration based on friction
        max_accel_total = self.coeff_friction * self.gravity
        
        # Calculate magnitude of desired total acceleration
        accel_total_desired = np.sqrt(accel_lon_desired**2 + accel_lat_desired**2)
        
        # Scale down accelerations if exceeding grip limit
        if accel_total_desired > max_accel_total:
            scale_factor = max_accel_total / accel_total_desired
            self.accel_lon = accel_lon_desired * scale_factor
            self.accel_lat = accel_lat_desired * scale_factor
        else:
            self.accel_lon = accel_lon_desired
            self.accel_lat = accel_lat_desired
        
        # --- State Update (Euler Integration) --- 
        # Velocity update
        v_dot = self.accel_lon
        self.v += v_dot * dt
        self.v = np.clip(self.v, 0, self.max_v) # Ensure non-negative and cap speed
        
        # Heading update (theta_dot derived from lateral acceleration)
        # a_lat = v * theta_dot  => theta_dot = a_lat / v (avoid division by zero)
        if abs(self.v) > 0.1:
            theta_dot = self.accel_lat / self.v 
        else:
            theta_dot = 0.0
            
        self.theta += theta_dot * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi # Normalize angle
        
        # Position update
        x_dot = self.v * np.cos(self.theta)
        y_dot = self.v * np.sin(self.theta)
        self.last_x = self.x
        self.last_y = self.y
        self.x += x_dot * dt
        self.y += y_dot * dt
        
    def get_data(self):
        """Returns a dictionary containing the car's current state values."""
        return {
            'x': self.x,
            'y': self.y,
            'v': self.v,
            'theta': self.theta,
            'steer_angle': self.steer_angle,
            'accel': self.accel_lon, # Report longitudinal acceleration
            'accel_lat': self.accel_lat # Report lateral acceleration
            # Add other relevant raw car data here if needed later
        }

    # The draw method previously here has been removed as it was unused
    # and the drawing logic is now in ui.py._draw_cars
