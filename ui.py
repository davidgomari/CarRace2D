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
        
        # Sidebar Configuration
        self.sidebar_width = 250  # Width of the sidebar in pixels
        self.sidebar_color = (50, 50, 50)  # Dark grey background
        self.sidebar_text_color = (220, 220, 220)  # Light grey text
        self.font_size = 18
        self.line_height = 22
        
        # Bottom bar configuration
        self.bottom_bar_height = 40  # Height of the bottom bar in pixels
        self.bottom_bar_color = (40, 40, 40)  # Slightly darker than sidebar
        self.back_button_width = 100
        self.back_button_height = 30
        self.back_button_color = (70, 70, 70)
        self.back_button_hover_color = (100, 100, 100)
        self.back_button_text_color = (220, 220, 220)
        
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
        
        # Back button rectangle
        self.back_button_rect = pygame.Rect(
            self.screen_width - self.back_button_width - 10,  # 10px padding from right
            sim_area_height + (self.bottom_bar_height - self.back_button_height) // 2,  # Centered vertically
            self.back_button_width,
            self.back_button_height
        )
        
        # Adjust world offset for the simulation area
        self.world_offset_x = sim_area_width / 2
        self.world_offset_y = sim_area_height / 2
        
        self.screen = None
        self.clock = None
        self.font = None
        
        if self.render_mode == 'human':
            self._init_pygame()
    
    def _init_pygame(self):
        """Initialize pygame for human rendering mode."""
        pygame.init()
        pygame.font.init()  # Initialize font module
        pygame.display.set_caption("Racing Simulation")
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
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
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                import sys
                sys.exit()
        
        # Clear simulation area (lighter grey)
        self.screen.fill((200, 200, 200), self.sim_area_rect)
        # Clear sidebar area (darker grey)
        self.screen.fill(self.sidebar_color, self.sidebar_rect)
        
        # Draw track and cars in the simulation area
        self._draw_track_and_cars(cars)
        # Draw the sidebar content
        self._draw_sidebar(cars, current_step, dt)
        
        # Draw the bottom bar with back button
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
        temp_surface.fill((200, 200, 200))  # Background color
        
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
    
    def _draw_track(self):
        """Draws the track onto the main screen's sim area."""
        screen = self.screen
        world_to_screen = self.world_to_screen
        track = self.track # Access the track object stored in the UI
        
        # --- Draw Track Boundaries --- 
        outer_r = track.radius + track.half_width
        inner_r = track.radius - track.half_width
        
        # Straight sections (outer)
        pygame.draw.line(screen, (0, 0, 0), 
                        world_to_screen(-track.length/2, outer_r),
                        world_to_screen(track.length/2, outer_r), 2)
        pygame.draw.line(screen, (0, 0, 0),
                        world_to_screen(-track.length/2, -outer_r),
                        world_to_screen(track.length/2, -outer_r), 2)
        
        # Curved sections (outer)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        points = [(track.curve1_center_x + outer_r * np.cos(t), 
                  track.curve1_center_y + outer_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(screen, (0, 0, 0), False, screen_points, 2)
        
        theta = np.linspace(np.pi/2, 3*np.pi/2, 100)
        points = [(track.curve2_center_x + outer_r * np.cos(t),
                  track.curve2_center_y + outer_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(screen, (0, 0, 0), False, screen_points, 2)
        
        # Straight sections (inner)
        pygame.draw.line(screen, (0, 0, 0),
                        world_to_screen(-track.length/2, inner_r),
                        world_to_screen(track.length/2, inner_r), 2)
        pygame.draw.line(screen, (0, 0, 0),
                        world_to_screen(-track.length/2, -inner_r),
                        world_to_screen(track.length/2, -inner_r), 2)
        
        # Curved sections (inner)
        theta = np.linspace(-np.pi/2, np.pi/2, 100)
        points = [(track.curve1_center_x + inner_r * np.cos(t),
                  track.curve1_center_y + inner_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(screen, (0, 0, 0), False, screen_points, 2)
        
        theta = np.linspace(np.pi/2, 3*np.pi/2, 100)
        points = [(track.curve2_center_x + inner_r * np.cos(t),
                  track.curve2_center_y + inner_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(screen, (0, 0, 0), False, screen_points, 2)
        
        # --- Draw Centerline (Optional, using a different color) --- 
        center_r = track.radius
        centerline_color = (100, 100, 100) # Grey color for centerline
        # Straight centerlines
        pygame.draw.line(screen, centerline_color,
                        world_to_screen(-track.length/2, center_r),
                        world_to_screen(track.length/2, center_r), 1)
        pygame.draw.line(screen, centerline_color,
                        world_to_screen(-track.length/2, -center_r),
                        world_to_screen(track.length/2, -center_r), 1)
        
        # Curved centerlines
        theta = np.linspace(-np.pi/2, np.pi/2, 50) # Fewer points for dashed appearance if needed
        points = [(track.curve1_center_x + center_r * np.cos(t),
                  track.curve1_center_y + center_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(screen, centerline_color, False, screen_points, 1)
        
        theta = np.linspace(np.pi/2, 3*np.pi/2, 50)
        points = [(track.curve2_center_x + center_r * np.cos(t),
                  track.curve2_center_y + center_r * np.sin(t)) for t in theta]
        screen_points = [world_to_screen(x, y) for x, y in points]
        pygame.draw.lines(screen, centerline_color, False, screen_points, 1)
        
        # --- Draw Start/Finish Line --- 
        start_finish_color = (0, 0, 255) # Blue color
        yc_start_line = track.radius
        if track.start_lane == 'bottom':
            yc_start_line *= -1
        pygame.draw.line(screen, start_finish_color,
                        world_to_screen(track.start_line_x, yc_start_line - track.half_width),
                        world_to_screen(track.start_line_x, yc_start_line + track.half_width), 3)
    
    def _draw_track_and_cars(self, cars):
        """Draw the track and all cars onto the main screen's sim area."""
        self._draw_track() # Call the new method to draw the track
        self._draw_cars(self.screen, self.world_to_screen, cars, draw_info_text=False)
    
    def _draw_cars(self, surface, world_to_screen, cars, draw_info_text=False):
        """Draw all cars onto the specified surface."""
        for agent_id, car in cars.items():
            # Calculate car center screen coordinates once
            center_sx, center_sy = world_to_screen(car.x, car.y)
            
            # Ensure drawing is within the simulation area bounds (simple clipping based on center)
            if not self.sim_area_rect.collidepoint(center_sx, center_sy):
                continue  # Skip drawing if center is outside sim area
            
            # --- Draw Car Body (Rectangle) --- 
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
            
            # Draw the car polygon
            pygame.draw.polygon(surface, pygame.Color(car.color), screen_points)
            pygame.draw.polygon(surface, (0,0,0), screen_points, 1) # Black outline
            
            # --- Draw Heading Line --- 
            # Calculate end point of the heading line in world coordinates
            hx = car.x + car_length/2 * cos_theta 
            hy = car.y + car_length/2 * sin_theta
            # Convert end point to screen coordinates
            shx, shy = world_to_screen(hx, hy)
            # Draw the line from center to front
            pygame.draw.line(surface, (0, 0, 0), (center_sx, center_sy), (shx, shy), 2) # Black heading line
    
    def _draw_sidebar(self, cars, current_step, dt):
        """Draws the information sidebar onto the main screen."""
        if not self.font:
            return
        
        start_x = self.sim_area_rect.width + 10  # Start 10px into the sidebar
        current_y = 10
        
        # --- Simulation Info ---
        sim_time = current_step * dt
        time_text = self.font.render(f"Time: {sim_time:.2f} s", True, self.sidebar_text_color)
        self.screen.blit(time_text, (start_x, current_y))
        current_y += self.line_height
        
        steps_text = self.font.render(f"Step: {current_step}", True, self.sidebar_text_color)
        self.screen.blit(steps_text, (start_x, current_y))
        current_y += self.line_height * 1.5  # Add extra space
        
        # --- Agent Info ---
        title_text = self.font.render("Agent Information:", True, self.sidebar_text_color)
        self.screen.blit(title_text, (start_x, current_y))
        current_y += self.line_height * 1.2
        
        # Sort agent IDs for consistent display order
        sorted_agent_ids = sorted(cars.keys())
        
        for agent_id in sorted_agent_ids:
            car = cars.get(agent_id)
            if not car:
                continue
            
            # Agent ID and Color Indicator
            id_text = self.font.render(f"{agent_id}:", True, self.sidebar_text_color)
            id_rect = id_text.get_rect(topleft=(start_x + 25, current_y))
            pygame.draw.circle(self.screen, pygame.Color(car.color), (start_x + 10, current_y + self.font.get_height() // 2), 7)
            self.screen.blit(id_text, id_rect)
            current_y += self.line_height
            
            # Position
            pos_text = self.font.render(f"  Pos: ({car.x:.1f}, {car.y:.1f})", True, self.sidebar_text_color)
            self.screen.blit(pos_text, (start_x, current_y))
            current_y += self.line_height
            
            # Speed
            speed_text = self.font.render(f"  Speed: {car.v:.2f} m/s", True, self.sidebar_text_color)
            self.screen.blit(speed_text, (start_x, current_y))
            current_y += self.line_height
            
            # Lap Count
            lap_text = self.font.render(f"  Lap: {car.lap_count}", True, self.sidebar_text_color)
            self.screen.blit(lap_text, (start_x, current_y))
            current_y += self.line_height * 1.2  # Extra space between agents
    
    def _draw_bottom_bar(self):
        """Draw the bottom bar with the back button."""
        # Draw the bottom bar background
        pygame.draw.rect(self.screen, self.bottom_bar_color, self.bottom_bar_rect)
        
        # Draw the back button
        mouse_pos = pygame.mouse.get_pos()
        button_color = self.back_button_hover_color if self.back_button_rect.collidepoint(mouse_pos) else self.back_button_color
        pygame.draw.rect(self.screen, button_color, self.back_button_rect)
        pygame.draw.rect(self.screen, self.back_button_text_color, self.back_button_rect, 1)  # Border
        
        # Draw the back button text
        back_text = self.font.render("Back", True, self.back_button_text_color)
        text_rect = back_text.get_rect(center=self.back_button_rect.center)
        self.screen.blit(back_text, text_rect)
    
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
        return True
    
    def should_return_to_menu(self):
        """Check if the UI is requesting to return to the main menu."""
        return self.return_to_menu
    
    def close(self):
        """Close the environment and cleanup pygame."""
        if self.screen is not None:
            pygame.display.quit()
            pygame.font.quit()  # Quit font module
            pygame.quit()
            self.screen = None
            self.font = None
            self.clock = None 