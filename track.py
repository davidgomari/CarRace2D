# track.py
import numpy as np

class OvalTrack:
    def __init__(self, config):
        self.length = config['length']  # Straight section length
        self.radius = config['radius']  # Curve radius
        self.width = config['width']    # Total track width
        self.half_width = config['width'] / 2.0
        self.start_line_x = config['start_line_x']
        self.start_lane = config['start_lane']

        # Define key geometric points (centers of curves, etc.)
        self.curve1_center_x = self.length / 2.0
        self.curve1_center_y = 0.0
        self.curve2_center_x = -self.length / 2.0
        self.curve2_center_y = 0.0

        # Calculate total centerline length (approximate for RL reward shaping)
        self.centerline_length = 2 * self.length + 2 * np.pi * self.radius

        # Calculate inner and outer radius
        self.inner_radius = self.radius - self.width / 2.0
        self.outer_radius = self.radius + self.width / 2.0

    def __calculate_relative_angle_change_in_curve__(self, x, y):
        """Calculate angle relative to curve center for points in curved sections."""
        if x > self.length/2:  # Right curve
            dx = x - self.curve1_center_x
            dy = y - self.curve1_center_y
            return np.arctan2(dy, dx)
        elif x < -self.length/2:  # Left curve
            dx = x - self.curve2_center_x
            dy = y - self.curve2_center_y
            return np.arctan2(dy, dx)
        else:  # Straight section - shouldn't be called for these
            return 0.0
        
    def get_oval_centerline_point(self, s_raw):
        """
        Calculates the (x, y, theta) coordinates and tangent angle 
        on the oval track's centerline for a given arc length s_raw.
        theta: represents the direction you would be facing if you were
               driving along the centerline at the point (x,y)
        Assumes s=0 starts at the beginning of the first straight segment.
            s_raw in [0, total_length)
        """
        L = self.length
        R = self.radius
        total_length = self.centerline_length
        s = s_raw % total_length # Ensure s is within [0, total_length)

        # Define critical points (arc lengths at segment transitions)
        s_bottom_straight_end = L
        s_right_curve_end = L + np.pi * R
        s_top_straight_end = L + np.pi * R + L
        # s_left_curve_end = total_length

        if 0 <= s < s_bottom_straight_end: # Bottom straight
            prog_in_seg = s
            x = -L/2 + prog_in_seg
            y = -R
            theta = 0.0
        elif s_bottom_straight_end <= s < s_right_curve_end: # Right curve
            prog_in_seg = s - s_bottom_straight_end
            angle_in_curve = prog_in_seg / R 
            # Curve starts at (-pi/2) relative to curve center's +x axis
            current_angle_on_circle = -np.pi/2 + angle_in_curve 
            x = self.curve1_center_x + R * np.cos(current_angle_on_circle)
            y = self.curve1_center_y + R * np.sin(current_angle_on_circle)
            theta = current_angle_on_circle + np.pi/2
        elif s_right_curve_end <= s < s_top_straight_end: # Top straight
            prog_in_seg = s - s_right_curve_end
            x = L/2 - prog_in_seg
            y = R
            theta = np.pi
        else: # Left curve (s_top_straight_end <= s < total_length)
            prog_in_seg = s - s_top_straight_end
            angle_in_curve = prog_in_seg / R
            # Curve starts at (pi/2) relative to curve center's +x axis
            current_angle_on_circle = np.pi/2 + angle_in_curve
            x = self.curve2_center_x + R * np.cos(current_angle_on_circle)
            y = self.curve2_center_y + R * np.sin(current_angle_on_circle)
            theta = current_angle_on_circle + np.pi/2

        theta = (theta + np.pi) % (2 * np.pi) - np.pi # Normalize theta to [-pi, pi]
        return x, y, theta    

    def calculate_progress(self, last_x, last_y, x, y):
        cond_x_in_middle      = (-self.length/2 <= x <= self.length/2)
        cond_last_x_in_middle = (-self.length/2 <= last_x <= self.length/2)
        
        if cond_x_in_middle and cond_last_x_in_middle:
            # Both points in straight section
            delta_progress = abs(last_x - x) / self.centerline_length
            return delta_progress
        elif (x > self.length/2 and last_x > self.length/2) or (x < -self.length/2 and last_x < -self.length/2):
            # Points are in the same curve
            theta_last = self.__calculate_relative_angle_change_in_curve__(last_x, last_y)
            theta = self.__calculate_relative_angle_change_in_curve__(x, y)
            delta_angle = (theta - theta_last) % (2 * np.pi)
            if delta_angle > np.pi:
                delta_angle = 2 * np.pi - delta_angle
            delta_progress = delta_angle * self.radius / self.centerline_length
            return delta_progress
        else:
            # Points are in different curves - approximate using straight-line distance
            delta_progress = np.sqrt((x - last_x)**2 + (y - last_y)**2) / self.centerline_length
            return delta_progress

    def is_on_track(self, x, y):
        """ Checks if point (x, y) is within the track boundaries. (OK)"""
        # Check straight sections
        if -self.length / 2 <= x <= self.length / 2:
            if self.inner_radius <= abs(y) <= self.outer_radius:
                return True
        # Check curved sections
        elif x > self.length / 2: # Right curve
            dist_sq = (x - self.curve1_center_x)**2 + (y - self.curve1_center_y)**2
            if self.inner_radius**2 <= dist_sq <= self.outer_radius**2:
                 return True
        elif x < -self.length / 2: # Left curve
            dist_sq = (x - self.curve2_center_x)**2 + (y - self.curve2_center_y)**2
            if self.inner_radius**2 <= dist_sq <= self.outer_radius**2:
                 return True
        return False

    def vectorized_is_on_track(self, x_coords, y_coords):
        """Vectorized version of track boundary check for multiple points.
        
        Args:
            x_coords (array-like): Array of x-coordinates
            y_coords (array-like): Array of y-coordinates
            
        Returns:
            numpy.ndarray: Boolean array indicating which points are on track
        """
        # Convert to numpy arrays if they aren't already
        x_coords = np.asarray(x_coords)
        y_coords = np.asarray(y_coords)
        
        # Initialize result array
        result = np.zeros(len(x_coords), dtype=bool)
        
        # Check straight sections
        straight_mask = (-self.length / 2 <= x_coords) & (x_coords <= self.length / 2)
        straight_valid = (self.inner_radius <= np.abs(y_coords)) & (np.abs(y_coords) <= self.outer_radius)
        result[straight_mask] = straight_valid[straight_mask]
        
        # Check right curve
        right_curve_mask = x_coords > self.length / 2
        if np.any(right_curve_mask):
            dx_right = x_coords[right_curve_mask] - self.curve1_center_x
            dy_right = y_coords[right_curve_mask] - self.curve1_center_y
            dist_sq_right = dx_right**2 + dy_right**2
            right_valid = (self.inner_radius**2 <= dist_sq_right) & (dist_sq_right <= self.outer_radius**2)
            result[right_curve_mask] = right_valid
        
        # Check left curve
        left_curve_mask = x_coords < -self.length / 2
        if np.any(left_curve_mask):
            dx_left = x_coords[left_curve_mask] - self.curve2_center_x
            dy_left = y_coords[left_curve_mask] - self.curve2_center_y
            dist_sq_left = dx_left**2 + dy_left**2
            left_valid = (self.inner_radius**2 <= dist_sq_left) & (dist_sq_left <= self.outer_radius**2)
            result[left_curve_mask] = left_valid
        
        return result

    def get_distance_to_centerline(self, x, y):
        """Calculate the distance from a point to the centerline."""
        # For straight sections
        if -self.length / 2 <= x <= self.length / 2:
            return abs(y) - self.radius
        
        # For curved sections
        if x > self.length / 2:  # Right curve
            center_x, center_y = self.curve1_center_x, self.curve1_center_y
        else:  # Left curve
            center_x, center_y = self.curve2_center_x, self.curve2_center_y
            
        # Calculate distance to center of curve
        dx = x - center_x
        dy = y - center_y
        dist_to_center = np.sqrt(dx*dx + dy*dy)
        
        # Distance to centerline is difference between distance to center and radius
        return dist_to_center - self.radius

    def check_lap_completion(self, car_x, car_last_x, car_y):
        """ Checks if the car crossed the start/finish line. """
        
        # Check if the car **crossed** the start line
        if self.start_lane == 'bottom' and (car_last_x < self.start_line_x <= car_x) and car_y < 0:
            crossed = True
        elif self.start_lane == 'top' and (car_last_x > self.start_line_x >= car_x) and car_y > 0:
            crossed = True
        else:
            crossed = False
        
        # Ensure the car is within track bounds in the y-direction
        # Since start line is on straight section, check if y is between inner and outer radius
        within_bounds = self.inner_radius <= abs(car_y) <= self.outer_radius

        return crossed and within_bounds

    def get_starting_positions(self, num_agents):
        """ Defines starting positions based on the image. (OK)"""
        # Image shows cars staggered slightly behind the start line
        # Position 1 (Blue): Center, slightly behind line
        # Position 2 (Green): Inner lane, further behind
        # Position 3 (Red): Outer lane, further behind

        distance_between_cars = 6.0
        scaler = 1.0 # 1.0 for top lane, -1.0 for bottom lane
        if self.start_lane == 'bottom':
            scaler = -1.0

        positions = []
        base_x = self.start_line_x + scaler * 3.0 # Start slightly behind the line

        # Pattern: Fill
        for i in range(num_agents):
            x = base_x + scaler * distance_between_cars * (i // 2)
            y = self.radius * scaler + 0.5 * self.half_width * ((-1)**i)
            theta = 0.0     # Start facing forward
            v = 0.0         # Start stationary
            positions.append(np.array([x, y, v, theta], dtype=np.float32))

        return positions

    def is_correct_direction(self, x, y, theta):
        """Check if the car is moving in the correct direction (counterclockwise).
        
        Args:
            x (float): Car's x position
            y (float): Car's y position
            theta (float): Car's heading angle in radians
        
        Returns:
            bool: True if car is moving in correct direction, False otherwise
        """
        # Normalize theta to [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        # For straight sections
        if -self.length / 2 <= x <= self.length / 2:
            if y < 0:  # Bottom straight
                # Should be moving right (theta close to 0)
                return abs(theta) < np.pi/2
            else:  # Top straight
                # Should be moving left (theta close to pi or -pi)
                return abs(abs(theta) - np.pi) < np.pi/2
        
        # For curved sections
        elif x > self.length / 2:  # Right curve
            # Calculate angle to center of right curve
            dx = x - self.curve1_center_x
            dy = y - self.curve1_center_y
            angle_to_center = np.arctan2(dy, dx)
            # Desired heading should be perpendicular to angle_to_center + pi/2
            desired_theta = angle_to_center + np.pi/2
            # Normalize the difference
            theta_diff = abs((theta - desired_theta + np.pi) % (2 * np.pi) - np.pi)
            return theta_diff < np.pi/2
            
        else:  # Left curve
            # Calculate angle to center of left curve
            dx = x - self.curve2_center_x
            dy = y - self.curve2_center_y
            angle_to_center = np.arctan2(dy, dx)
            # Desired heading should be perpendicular to angle_to_center + pi/2
            desired_theta = angle_to_center + np.pi/2
            # Normalize the difference
            theta_diff = abs((theta - desired_theta + np.pi) % (2 * np.pi) - np.pi)
            return theta_diff < np.pi/2
        
    def get_curvature_at_s(self, s_raw):
        """Approximates centerline curvature at arc length s_raw."""
        ds = 0.5  # Slightly larger ds for stability in theta calculation
        # Ensure s_raw is not at the very start or end if using ds/2 offsets without wrapping s
        total_len = self.centerline_length
        s1 = (s_raw - ds / 2 + total_len) % total_len # Handle wrapping
        s2 = (s_raw + ds / 2) % total_len

        _, _, theta1 = self.get_oval_centerline_point(s1)
        _, _, theta2 = self.get_oval_centerline_point(s2)

        dtheta = theta2 - theta1
        # Normalize angle difference to [-pi, pi]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

        curvature = dtheta / ds if abs(ds) > 1e-6 else 0.0
        return curvature
    
    def get_racing_line_point(self, s, opponent_positions=None):
        """
        Calculates the (x, y, theta, v_ref) coordinates on the racing line for a given arc length s,
        adjusted for opponent positions to avoid collisions.

        Args:
            s (float): Arc length along the track in meters.
            opponent_positions (list): List of dictionaries with opponent data, each containing
                {'x': float, 'y': float, 'radius': float}. If None, uses base racing line.

        Returns:
            tuple: (x, y, theta, v_ref) where
                x (float): x-coordinate in meters
                y (float): y-coordinate in meters
                theta (float): Heading angle in radians, normalized to [-pi, pi]
                v_ref (float): Reference speed in m/s
        """
        # Track parameters
        L = self.length
        R = self.radius
        total_length = self.centerline_length
        s = s % total_length  # Wrap around track length

        # Safety margin from track edges
        margin = 1.0
        half_width = self.half_width
        min_radius = self.inner_radius + margin
        max_radius = self.outer_radius - margin

        # Define base racing line offset (closer to inner edge in curves)
        def get_base_offset(s_segment):
            # Smoothly vary offset: 0 (centerline) on straights, -0.8*half_width in curves
            if 0 <= s_segment < L or L + np.pi * R <= s_segment < 2 * L + np.pi * R:
                return 0.0  # Centerline on straights
            else:
                # Smooth transition into/out of curves
                if s_segment < L:
                    t = s_segment / (L * 0.1)  # Transition over 10% of straight
                elif s_segment < L + np.pi * R:
                    t = (s_segment - (L + np.pi * R * 0.9)) / (np.pi * R * 0.1)
                else:
                    t = (s_segment - (2 * L + np.pi * R)) / (np.pi * R * 0.1)
                t = min(max(t, 0.0), 1.0)
                return -0.8 * half_width * np.sin(np.pi * t / 2)  # Smooth offset

        # Base centerline coordinates
        x_base, y_base, theta_base = self.get_oval_centerline_point(s)

        # Adjust for racing line
        offset = get_base_offset(s)
        if 0 <= s < L:  # Bottom straight
            x = x_base
            y = y_base + offset
            theta = theta_base
        elif L <= s < L + np.pi * R:  # Right curve
            angle = (s - L) / R - np.pi / 2
            r = R + offset
            r = max(min(r, max_radius), min_radius)
            x = self.curve1_center_x + r * np.cos(angle)
            y = self.curve1_center_y + r * np.sin(angle)
            theta = angle + np.pi / 2
        elif L + np.pi * R <= s < 2 * L + np.pi * R:  # Top straight
            x = x_base
            y = y_base + offset
            theta = theta_base
        else:  # Left curve
            angle = (s - (2 * L + np.pi * R)) / R + np.pi / 2
            r = R + offset
            r = max(min(r, max_radius), min_radius)
            x = self.curve2_center_x + r * np.cos(angle)
            y = self.curve2_center_y + r * np.sin(angle)
            theta = angle + np.pi / 2

        # Collision avoidance adjustment
        if opponent_positions:
            min_dist = 2.0  # Minimum distance to maintain from opponents
            for opponent in opponent_positions:
                dx = x - opponent['x']
                dy = y - opponent['y']
                dist = np.sqrt(dx**2 + dy**2)
                sum_radii = opponent.get('radius', 1.5) + margin
                if dist < sum_radii + min_dist:
                    # Shift away from opponent
                    if 0 <= s < L or L + np.pi * R <= s < 2 * L + np.pi * R:  # Straights
                        shift = (sum_radii + min_dist - dist) * np.sign(dy)
                        y_new = y + shift
                        if abs(y_new) <= max_radius and abs(y_new) >= min_radius:
                            y = y_new
                    else:  # Curves
                        r_current = np.sqrt((x - self.curve1_center_x)**2 + y**2) if s < L + np.pi * R else np.sqrt((x - self.curve2_center_x)**2 + y**2)
                        shift = (sum_radii + min_dist - dist)
                        r_new = r_current + shift
                        if min_radius <= r_new <= max_radius:
                            angle = (s - L) / R - np.pi / 2 if s < L + np.pi * R else (s - (2 * L + np.pi * R)) / R + np.pi / 2
                            center_x = self.curve1_center_x if s < L + np.pi * R else self.curve2_center_x
                            center_y = self.curve2_center_y  # 0.0 for both curves
                            x = center_x + r_new * np.cos(angle)
                            y = center_y + r_new * np.sin(angle)
                            theta = angle + np.pi / 2

        # Normalize theta
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # Placeholder reference speed (adjust based on curvature and vehicle dynamics)
        curvature = self.get_curvature_at_s(s)
        mu = 1.1  # Friction coefficient (from car config)
        g = 9.81  # Gravity
        v_ref = np.sqrt(mu * g / max(abs(curvature), 1e-6)) if curvature != 0 else 20.0  # Cap at max speed
        v_ref = min(v_ref, 20.0)  # Respect max_speed from car config

        return x, y, theta, v_ref
