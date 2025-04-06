# track.py
import numpy as np

# TODO:
# - check_lap_completion: the agent can turn around in his place near the start lane without going throuth the whole track

class OvalTrack:
    # def __init__(self, length, radius, width, start_line_x=0.0, start_lane='bottom'):
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

    def get_centerline_point(self, progress_ratio):
        """ Gets a point on the centerline based on progress (0 to 1). """
        dist = progress_ratio * self.centerline_length
        # TODO: Implement logic to map distance to (x, y) on the oval centerline
        # This involves checking which segment (straight/curve) the distance falls into.
        x, y = 0, 0 # Placeholder
        theta = 0   # Placeholder (tangent angle)
        return x, y, theta

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
            return abs(abs(y) - self.radius)
        
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
        return abs(dist_to_center - self.radius)

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