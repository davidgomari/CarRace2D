import pygame
import numpy as np
import sys

class UI:
    def __init__(self, config, track):
        """Initialize the UI with configuration and track."""
        self.config = config
        self.track = track
        self.render_mode = config['simulation']['render_mode']
        self.render_fps = config['simulation']['render_fps']
        self.world_scale = 5.0  # Pixels per meter
        
        # Flag to indicate if we should return to the main menu
        self.return_to_menu = False
        self.should_reset = False
        
        # Performance settings
        self.use_simple_glow = True  # Use simpler glow effect for better performance
        self.glow_radius = 2  # Reduced from 3 to 2 for better performance
        self.glow_layers = 2  # Reduced number of glow layers
        
        # Sidebar Configuration - Cyberpunk style
        self.sidebar_width = 420  # Increased width for table
        self.sidebar_color = (15, 15, 20)  # Very dark background
        self.sidebar_text_color = (0, 255, 255)  # Cyan neon text
        self.header_text_color = (255, 255, 0) # Yellow for headers
        self.table_line_color = (50, 50, 60) # Dark grey for lines
        self.font_size = 24 # Slightly smaller for table
        self.line_height = 20 # Adjusted for smaller font
        
        # Load sidebar table columns from config
        self.sidebar_columns = config.get('ui', {}).get('sidebar_columns', [])
        if not self.sidebar_columns:
            print("Warning: UI sidebar_columns not found in config. Sidebar will be minimal.")
            # Provide default fallback columns if config is missing
            self.sidebar_columns = [
                { 'key': 'agent_id', 'header': 'Agent', 'format': '' },
                { 'key': 'v', 'header': 'Spd', 'format': '.1f' },
                { 'key': 'lap', 'header': 'Lap', 'format': 'd' }
            ]
            
        # Calculate column widths dynamically (simple equal distribution for now)
        self.column_widths = []
        if self.sidebar_columns:
            padding = 10 # Total padding within the sidebar
            available_width = self.sidebar_width - padding * 2
            base_width = available_width // len(self.sidebar_columns)
            remainder = available_width % len(self.sidebar_columns)
            self.column_widths = [base_width + (1 if i < remainder else 0) for i in range(len(self.sidebar_columns))] 

        # Bottom bar configuration - Cyberpunk style
        self.bottom_bar_height = 50  # Increased height for more buttons
        self.bottom_bar_color = (10, 10, 15)  # Even darker than sidebar
        self.button_width = 120
        self.button_height = 35
        self.button_color = (30, 30, 40)  # Darker button background
        self.button_hover_color = (50, 50, 60)  # Slightly lighter on hover
        self.button_text_color = (0, 255, 255)  # Cyan neon text
        self.button_border_color = (0, 200, 200)  # Cyan border
        
        # Track colors for cyberpunk style
        self.track_asphalt_color = (20, 20, 25)  # Very dark asphalt
        self.track_line_color = (0, 255, 255)  # Cyan neon lines
        self.track_curb_color = (255, 0, 128)  # Hot pink curbs
        self.start_finish_color = (0, 255, 128)  # Neon green for start/finish line
        
        # Neon glow effect parameters
        self.neon_glow_radius = self.glow_radius
        self.neon_glow_color = (0, 255, 255, 128)  # Semi-transparent cyan
        
        # Pre-rendered surfaces for better performance
        self.glow_surfaces = {}
        
        # Calculate simulation area dimensions
        track_cfg = config['track']
        sim_area_width = int((track_cfg['length'] + 2 * track_cfg['radius'] + 2*track_cfg['width']) * self.world_scale + 50)
        sim_area_height = int((2 * track_cfg['radius'] + 2 * track_cfg['width']) * self.world_scale + 50)
        
        # Total screen dimensions including sidebar and bottom bar
        self.screen_width = sim_area_width + self.sidebar_width
        self.screen_height = sim_area_height + self.bottom_bar_height
        
        # Simulation area rectangle (for clearing and drawing)
        self.sim_area_rect = pygame.Rect(0, 0, sim_area_width, sim_area_height)
        self.sidebar_rect = pygame.Rect(sim_area_width, 0, self.sidebar_width, sim_area_height)
        self.bottom_bar_rect = pygame.Rect(0, sim_area_height, self.screen_width, self.bottom_bar_height)
        
        # Button rectangles
        button_spacing = 10
        total_buttons_width = 3 * self.button_width + 2 * button_spacing
        start_x = (self.screen_width - total_buttons_width) // 2
        
        self.back_button_rect = pygame.Rect(
            start_x,
            sim_area_height + (self.bottom_bar_height - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        
        self.pause_button_rect = pygame.Rect(
            start_x + self.button_width + button_spacing,
            sim_area_height + (self.bottom_bar_height - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        
        self.reset_button_rect = pygame.Rect(
            start_x + 2 * (self.button_width + button_spacing),
            sim_area_height + (self.bottom_bar_height - self.button_height) // 2,
            self.button_width,
            self.button_height
        )
        
        # Adjust world offset for the simulation area
        self.world_offset_x = sim_area_width / 2
        self.world_offset_y = sim_area_height / 2
        
        self.screen = None
        self.clock = None
        self.font = None
        
        # Pause state
        self.paused = False
        
        if self.render_mode == 'human':
            self._init_pygame()
    
    def _init_pygame(self):
        """Initialize pygame for human rendering mode."""
        pygame.init()
        pygame.font.init()  # Initialize font module
        pygame.display.set_caption("Racing Simulation")
        
        # Try to use hardware acceleration if available
        try:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            print("Using hardware acceleration for rendering")
        except:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            print("Hardware acceleration not available, using software rendering")
            
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.SysFont(None, self.font_size)  # Use system default font
        except Exception as e:
            print(f"Warning: Could not load system font. Using default pygame font. Error: {e}")
            self.font = pygame.font.Font(None, self.font_size)  # Fallback pygame font
        if self.font is None:  # Final fallback
            print("Error: Pygame font initialization failed completely.")
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        sx = int(self.world_offset_x + x * self.world_scale)
        sy = int(self.world_offset_y - y * self.world_scale)
        return sx, sy
    
    def render(self, cars, current_step, dt):
        """Render the environment."""
        if self.render_mode != 'human':
            if self.render_mode == "rgb_array":
                return self._render_rgb_array(cars)
            return
        
        if self.screen is None:
            self._init_pygame()
            if self.screen is None:
                print("Error: Pygame screen not initialized. Cannot render.")
                return
        
        self._render_human(cars, current_step, dt)
    
    def _render_human(self, cars, current_step, dt):
        """Render the environment in human mode with sidebar."""
        if self.font is None:
            print("Warning: Font not available, cannot render text.")
        
        # Handle UI events only in handle_events method
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         self.close()
        #         import sys
        #         sys.exit()
        
        # Clear simulation area (asphalt color)
        self.screen.fill(self.track_asphalt_color, self.sim_area_rect)
        # Clear sidebar area (darker grey)
        self.screen.fill(self.sidebar_color, self.sidebar_rect)
        
        # Draw track and cars in the simulation area
        self._draw_track_and_cars(cars)
        # Draw the sidebar content
        self._draw_sidebar(cars, current_step, dt)
        
        # Draw the bottom bar with buttons
        self._draw_bottom_bar()
        
        pygame.display.flip()
        if self.clock:
            self.clock.tick(self.render_fps)
    
    def _render_rgb_array(self, cars):
        """Render the environment in rgb_array mode (WITHOUT sidebar)."""
        sim_area_width = self.sim_area_rect.width
        sim_area_height = self.sim_area_rect.height
        
        # Create a temporary surface matching the sim area size
        temp_surface = pygame.Surface((sim_area_width, sim_area_height))
        temp_surface.fill(self.track_asphalt_color)  # Background color
        
        # Need a world_to_screen function specifically for this surface
        def temp_world_to_screen(x, y):
            sx = int(sim_area_width / 2 + x * self.world_scale)
            sy = int(sim_area_height / 2 - y * self.world_scale)
            return sx, sy
        
        # Draw track and cars onto the temporary surface
        self._draw_track(temp_surface)
        self._draw_cars(temp_surface, temp_world_to_screen, cars, draw_info_text=False)
        
        # Get image data from the temporary surface
        return np.frombuffer(pygame.image.tostring(temp_surface, "RGB"), dtype=np.uint8).reshape(sim_area_height, sim_area_width, 3)
    
    def _draw_track(self, surface=None):
        """Draws the track onto the specified surface or main screen's sim area with cyberpunk style."""
        screen = surface if surface else self.screen
        world_to_screen = self.world_to_screen
        track = self.track # Access the track object stored in the UI
        
        # --- Draw Track Boundaries --- 
        outer_r = track.radius + track.half_width
        inner_r = track.radius - track.half_width
        
        # Draw curbs (outer) with neon glow effect
        # Straight sections (outer)
        self._draw_neon_line(screen, world_to_screen(-track.length/2, outer_r),
                            world_to_screen(track.length/2, outer_r), self.track_curb_color, 4)
        self._draw_neon_line(screen, world_to_screen(-track.length/2, -outer_r),
                            world_to_screen(track.length/2, -outer_r), self.track_curb_color, 4)
        
        # Curved sections (outer)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        points = [(track.curve1_center_x + outer_r * np.cos(t), 
                  track.curve1_center_y + outer_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        self._draw_neon_lines(screen, screen_points, self.track_curb_color, 4)
        
        theta = np.linspace(np.pi/2, 3*np.pi/2, 100)
        points = [(track.curve2_center_x + outer_r * np.cos(t),
                  track.curve2_center_y + outer_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        self._draw_neon_lines(screen, screen_points, self.track_curb_color, 4)
        
        # Draw curbs (inner) with neon glow effect
        # Straight sections (inner)
        self._draw_neon_line(screen, world_to_screen(-track.length/2, inner_r),
                            world_to_screen(track.length/2, inner_r), self.track_curb_color, 4)
        self._draw_neon_line(screen, world_to_screen(-track.length/2, -inner_r),
                            world_to_screen(track.length/2, -inner_r), self.track_curb_color, 4)
        
        # Curved sections (inner)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        points = [(track.curve1_center_x + inner_r * np.cos(t),
                  track.curve1_center_y + inner_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        self._draw_neon_lines(screen, screen_points, self.track_curb_color, 4)
        
        theta = np.linspace(np.pi/2, 3*np.pi/2, 100)
        points = [(track.curve2_center_x + inner_r * np.cos(t),
                  track.curve2_center_y + inner_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        self._draw_neon_lines(screen, screen_points, self.track_curb_color, 4)
        
        # --- Draw Track Lines (Cyan Neon) --- 
        center_r = track.radius
        # Straight centerlines
        self._draw_neon_line(screen, world_to_screen(-track.length/2, center_r),
                            world_to_screen(track.length/2, center_r), self.track_line_color, 2)
        self._draw_neon_line(screen, world_to_screen(-track.length/2, -center_r),
                            world_to_screen(track.length/2, -center_r), self.track_line_color, 2)
        
        # Curved centerlines
        theta = np.linspace(-np.pi/2, np.pi/2, 50)
        points = [(track.curve1_center_x + center_r * np.cos(t),
                  track.curve1_center_y + center_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        self._draw_neon_lines(screen, screen_points, self.track_line_color, 2)
        
        theta = np.linspace(np.pi/2, 3*np.pi/2, 50)
        points = [(track.curve2_center_x + center_r * np.cos(t),
                  track.curve2_center_y + center_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        self._draw_neon_lines(screen, screen_points, self.track_line_color, 2)
        
        # --- Draw Start/Finish Line (Neon Green) --- 
        yc_start_line = track.radius
        if track.start_lane == 'bottom':
            yc_start_line *= -1
        self._draw_neon_line(screen, world_to_screen(track.start_line_x, yc_start_line - track.half_width),
                            world_to_screen(track.start_line_x, yc_start_line + track.half_width), 
                            self.start_finish_color, 5)
    
    def _draw_neon_line(self, surface, start_pos, end_pos, color, width):
        """Draw a line with a neon glow effect (optimized version)."""
        if self.use_simple_glow:
            # Simplified glow effect - just draw a few lines with decreasing width
            for i in range(self.glow_layers, 0, -1):
                glow_width = width + i
                pygame.draw.line(surface, color, start_pos, end_pos, glow_width)
            # Draw the main line
            pygame.draw.line(surface, color, start_pos, end_pos, width)
        else:
            # Original glow effect with alpha blending (more expensive)
            # Draw the glow effect (multiple lines with decreasing alpha)
            for i in range(self.neon_glow_radius, 0, -1):
                alpha = 128 // i
                glow_color = (*color[:3], alpha)
                glow_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
                pygame.draw.line(glow_surface, glow_color, start_pos, end_pos, width + i*2)
                surface.blit(glow_surface, (0, 0))
            
            # Draw the main line
            pygame.draw.line(surface, color, start_pos, end_pos, width)
    
    def _draw_neon_lines(self, surface, points, color, width):
        """Draw connected lines with a neon glow effect (optimized version)."""
        if self.use_simple_glow:
            # Simplified glow effect - just draw a few lines with decreasing width
            for i in range(self.glow_layers, 0, -1):
                glow_width = width + i
                pygame.draw.lines(surface, color, False, points, glow_width)
            # Draw the main lines
            pygame.draw.lines(surface, color, False, points, width)
        else:
            # Original glow effect with alpha blending (more expensive)
            # Draw the glow effect (multiple lines with decreasing alpha)
            for i in range(self.neon_glow_radius, 0, -1):
                alpha = 128 // i
                glow_color = (*color[:3], alpha)
                glow_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
                pygame.draw.lines(glow_surface, glow_color, False, points, width + i*2)
                surface.blit(glow_surface, (0, 0))
            
            # Draw the main lines
            pygame.draw.lines(surface, color, False, points, width)
    
    def _draw_track_and_cars(self, cars):
        """Draw the track and all cars onto the main screen's sim area."""
        self._draw_track() # Call the new method to draw the track
        self._draw_cars(self.screen, self.world_to_screen, cars, draw_info_text=False)
    
    def _draw_cars(self, surface, world_to_screen, cars, draw_info_text=False):
        """Draw all cars onto the specified surface with cyberpunk style (optimized)."""
        for agent_id, car in cars.items():
            # Calculate car center screen coordinates once
            center_sx, center_sy = world_to_screen(car.x, car.y)
            
            # Ensure drawing is within the simulation area bounds (simple clipping based on center)
            if not self.sim_area_rect.collidepoint(center_sx, center_sy):
                continue  # Skip drawing if center is outside sim area
            
            # --- Draw Car Body (Rectangle) with neon glow --- 
            car_length = car.params['length']
            car_width = car.params['width']
            # Define coordinates of the rectangle corners relative to the car center (0,0) before rotation
            car_outline_local = [
                (car_length/2, car_width/2),
                (-car_length/2, car_width/2),
                (-car_length/2, -car_width/2),
                (car_length/2, -car_width/2)
            ]
            
            # Rotate and translate points to world coordinates
            rotated_world_points = []
            cos_theta = np.cos(car.theta)
            sin_theta = np.sin(car.theta)
            for local_x, local_y in car_outline_local:
                world_x = car.x + local_x * cos_theta - local_y * sin_theta
                world_y = car.y + local_x * sin_theta + local_y * cos_theta
                rotated_world_points.append((world_x, world_y))
                
            # Convert world coordinates to screen coordinates
            screen_points = [world_to_screen(wx, wy) for wx, wy in rotated_world_points]
            
            # Draw the car with simplified glow effect
            if self.use_simple_glow:
                # Draw the glow effect with multiple polygons of increasing size
                for i in range(self.glow_layers, 0, -1):
                    # Create slightly larger polygon for glow
                    glow_points = []
                    for x, y in screen_points:
                        # Expand each point outward from the center
                        dx = x - center_sx
                        dy = y - center_sy
                        # Normalize and scale
                        length = (dx*dx + dy*dy)**0.5
                        if length > 0:
                            dx = dx / length * i
                            dy = dy / length * i
                        glow_points.append((x + dx, y + dy))
                    
                    # Draw the glow polygon
                    pygame.draw.polygon(surface, pygame.Color(car.color), glow_points)
                
                # Draw the main car polygon
                pygame.draw.polygon(surface, pygame.Color(car.color), screen_points)
                pygame.draw.polygon(surface, (0, 0, 0), screen_points, 1)  # Black outline
            else:
                # Original glow effect with alpha blending (more expensive)
                # Create a temporary surface for the glow effect
                glow_surface = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
                
                # Draw the glow effect
                for i in range(self.neon_glow_radius, 0, -1):
                    alpha = 128 // i
                    glow_color = (*pygame.Color(car.color)[:3], alpha)
                    pygame.draw.polygon(glow_surface, glow_color, screen_points)
                
                # Blit the glow surface onto the main surface
                surface.blit(glow_surface, (0, 0))
                
                # Draw the main car polygon
                pygame.draw.polygon(surface, pygame.Color(car.color), screen_points)
                pygame.draw.polygon(surface, (0, 0, 0), screen_points, 1)  # Black outline
            
            # Add a neon highlight to make the car look more cyberpunk
            highlight_points = screen_points[:2]  # Just the front edge
            self._draw_neon_line(surface, highlight_points[0], highlight_points[1], (255, 255, 255), 2)
            
            # --- Draw Heading Line with neon effect --- 
            # Calculate end point of the heading line in world coordinates
            hx = car.x + car_length/2 * cos_theta 
            hy = car.y + car_length/2 * sin_theta
            # Convert end point to screen coordinates
            shx, shy = world_to_screen(hx, hy)
            # Draw the line from center to front with neon effect
            self._draw_neon_line(surface, (center_sx, center_sy), (shx, shy), (0, 0, 0), 2)
            
            # Draw car ID above the car with cyberpunk style
            if draw_info_text and self.font:
                id_text = self.font.render(f"Car {agent_id}", True, self.sidebar_text_color)
                id_rect = id_text.get_rect(center=(center_sx, center_sy - 20))
                surface.blit(id_text, id_rect)
    
    def _draw_sidebar(self, cars, current_step, dt):
        """Draws the information sidebar with cyberpunk style (optimized)."""
        if not self.font:
            return
        
        padding = 10
        start_x = self.sim_area_rect.width + padding
        current_y = 10
        table_start_y = 50 # Leave space for sim info
        
        # --- Simulation Info --- (Keep this above the table)
        sim_time = current_step * dt
        time_text = self.font.render(f"Time: {sim_time:.2f} s", True, self.header_text_color) # Use header color
        self.screen.blit(time_text, (start_x, current_y))
        current_y += self.line_height
        
        steps_text = self.font.render(f"Step: {current_step}", True, self.header_text_color) # Use header color
        self.screen.blit(steps_text, (start_x, current_y))
        current_y = table_start_y # Reset Y for table
        
        # --- Draw Table Header --- 
        header_y = current_y
        col_x = start_x
        for i, col_info in enumerate(self.sidebar_columns):
            header_text = self.font.render(col_info['header'], True, self.header_text_color)
            # Center header text within the column width
            text_rect = header_text.get_rect(centerx=col_x + self.column_widths[i] // 2, top=header_y)
            self.screen.blit(header_text, text_rect)
            col_x += self.column_widths[i]
        
        current_y += self.line_height * 1.2 # Space after header
        # Draw header separator line
        pygame.draw.line(self.screen, self.table_line_color, 
                         (start_x - padding//2, current_y), 
                         (self.sim_area_rect.width + self.sidebar_width - padding//2, current_y), 1)
        current_y += padding // 2 # Space after line
        
        # --- Draw Table Rows --- 
        sorted_agent_ids = sorted(cars.keys())
        
        car_data_cache = {agent_id: car.get_data() for agent_id, car in cars.items()} # Cache car data
        
        for agent_id in sorted_agent_ids:
            car = cars.get(agent_id)
            if not car:
                continue
            
            row_y = current_y
            col_x = start_x
            car_data = car_data_cache.get(agent_id, {})

            for i, col_info in enumerate(self.sidebar_columns):
                key = col_info['key']
                fmt = col_info['format']
                value = "N/A" # Default value
                
                try:
                    if key == 'agent_id':
                        value = agent_id.split('_')[-1] # Show only the number
                    elif key == 'lap':
                        value = car.lap_count
                    elif key == 'agent_type':
                        # Agent type might need to be fetched from the main config
                        value = self.config.get('agents', {}).get(agent_id, {}).get('type', '?')[:3].upper() # Short type
                    elif key == 'dist_to_centerline':
                        # Calculate distance to centerline
                        value = self.track.get_distance_to_centerline(car.x, car.y)
                    elif key == 'dist_to_boundary':
                        # Calculate distance to boundary
                        dist_center = self.track.get_distance_to_centerline(car.x, car.y)
                        value = self.track.half_width - abs(dist_center)
                    elif key == 'total_progress':
                        value = car.total_progress
                    else:
                        # Try to get from cached car data
                        if key in car_data:
                             value = car_data[key]
                        # Optionally: Add fallbacks for other calculated values here
                             
                    # Format the value if possible
                    if value != "N/A" and fmt:
                        value = f"{value:{fmt}}"
                        
                except Exception as e:
                    # Keep "N/A" on error, maybe log e
                    # print(f"Sidebar Error fetching {key} for {agent_id}: {e}")
                    pass
                    
                # Render and blit the cell text (left-aligned for simplicity now)
                cell_text = self.font.render(str(value), True, self.sidebar_text_color)
                # Center text within the column width
                text_rect = cell_text.get_rect(centerx=col_x + self.column_widths[i] // 2, top=row_y)
                self.screen.blit(cell_text, text_rect)
                col_x += self.column_widths[i]
                
            current_y += self.line_height # Move to the next row
            # Optional: Draw row separator lines
            # pygame.draw.line(self.screen, self.table_line_color, 
            #                  (start_x - padding//2, current_y), 
            #                  (self.sim_area_rect.width + self.sidebar_width - padding//2, current_y), 1)
            # current_y += padding // 2
    
    def _draw_bottom_bar(self):
        """Draw the bottom bar with buttons in cyberpunk style."""
        # Draw the bottom bar background
        pygame.draw.rect(self.screen, self.bottom_bar_color, self.bottom_bar_rect)
        
        # Draw the buttons with cyberpunk style
        mouse_pos = pygame.mouse.get_pos()
        
        # Back button with neon effect
        button_color = self.button_hover_color if self.back_button_rect.collidepoint(mouse_pos) else self.button_color
        self._draw_neon_button(self.back_button_rect, button_color, "Back")
        
        # Pause/Resume button with neon effect
        button_color = self.button_hover_color if self.pause_button_rect.collidepoint(mouse_pos) else self.button_color
        self._draw_neon_button(self.pause_button_rect, button_color, "Pause" if not self.paused else "Resume")
        
        # Reset button with neon effect
        button_color = self.button_hover_color if self.reset_button_rect.collidepoint(mouse_pos) else self.button_color
        self._draw_neon_button(self.reset_button_rect, button_color, "Reset")
    
    def _draw_neon_button(self, rect, color, text):
        """Draw a button with a neon glow effect (optimized)."""
        if self.use_simple_glow:
            # Simplified glow effect - just draw multiple rectangles with decreasing size
            for i in range(self.glow_layers, 0, -1):
                glow_rect = pygame.Rect(
                    rect.x - i, rect.y - i,
                    rect.width + i*2, rect.height + i*2
                )
                pygame.draw.rect(self.screen, color, glow_rect, 0, 5)
            
            # Draw the main button
            pygame.draw.rect(self.screen, color, rect, 0, 5)
            pygame.draw.rect(self.screen, self.button_border_color, rect, 1, 5)  # Border
        else:
            # Original glow effect with alpha blending (more expensive)
            # Draw the glow effect
            glow_surface = pygame.Surface((rect.width + self.neon_glow_radius*2, rect.height + self.neon_glow_radius*2), pygame.SRCALPHA)
            for i in range(self.neon_glow_radius, 0, -1):
                alpha = 128 // i
                glow_color = (*color[:3], alpha)
                pygame.draw.rect(glow_surface, glow_color, (i, i, rect.width + (self.neon_glow_radius-i)*2, rect.height + (self.neon_glow_radius-i)*2), 0, 5)
            self.screen.blit(glow_surface, (rect.x - self.neon_glow_radius, rect.y - self.neon_glow_radius))
            
            # Draw the main button
            pygame.draw.rect(self.screen, color, rect, 0, 5)
            pygame.draw.rect(self.screen, self.button_border_color, rect, 1, 5)  # Border
        
        # Draw the text
        button_text = self.font.render(text, True, self.button_text_color)
        text_rect = button_text.get_rect(center=rect.center)
        self.screen.blit(button_text, text_rect)
    
    def handle_events(self):
        """Handle pygame events and return whether to continue running."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Detected QUIT event. Closing environment.")
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    if self.back_button_rect.collidepoint(event.pos):
                        print("Back button clicked. Returning to main menu.")
                        self.return_to_menu = True
                        return False
                    elif self.pause_button_rect.collidepoint(event.pos):
                        print("Pause/Resume button clicked.")
                        self.paused = not self.paused
                        return True
                    elif self.reset_button_rect.collidepoint(event.pos):
                        print("Reset button clicked.")
                        self.should_reset = True
                        return True
        return True
    
    def should_return_to_menu(self):
        """Check if the UI is requesting to return to the main menu."""
        return self.return_to_menu
    
    def should_reset(self):
        """Check if the UI is requesting to reset the environment."""
        if self.should_reset:
            self.should_reset = False
            return True
        return False
    
    def is_paused(self):
        """Check if the simulation is paused."""
        return self.paused
    
    def close(self):
        """Close the environment and cleanup pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()  # Quit font module
            pygame.quit()
            self.screen = None
            self.font = None
            self.clock = None