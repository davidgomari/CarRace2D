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

    def __to_centerline_point__(self, x, y):
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

    def calculate_progress(self, last_x, last_y, x, y):
        cond_x_in_middle      = (-self.length/2 <= x <= self.length/2)
        cond_last_x_in_middle = (-self.length/2 <= last_x <= self.length/2)
        
        if cond_x_in_middle and cond_last_x_in_middle:
            # Both points in straight section
            delta_progress = abs(last_x - x) / self.centerline_length
            return delta_progress
        elif (x > self.length/2 and last_x > self.length/2) or (x < -self.length/2 and last_x < -self.length/2):
            # Points are in the same curve
            theta_last = self.__to_centerline_point__(last_x, last_y)
            theta = self.__to_centerline_point__(x, y)
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