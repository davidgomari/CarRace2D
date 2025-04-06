import pygame
import sys
import yaml
import os

class MainMenu:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        # Screen dimensions
        self.width = 800
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racing Simulation")
        
        # Colors (Use RGBA for transparency)
        self.background_color = (50, 50, 50) # Fallback color
        self.text_color = (255, 255, 255)
        self.button_color = (70, 70, 70, 230)  # Added alpha (0-255, lower is more transparent)
        self.button_hover_color = (100, 100, 100, 210) # Slightly less transparent on hover
        self.button_text_color = (255, 255, 255)
        self.button_border_color = (220, 220, 220, 200) # Border color with alpha
        self.section_color = (60, 60, 60)
        
        # Load background image
        self.background_image = None
        self._load_background_image()
        
        # Fonts
        self.title_font = pygame.font.SysFont(None, 60) # Slightly larger title
        self.section_font = pygame.font.SysFont(None, 36)
        self.button_font = pygame.font.SysFont(None, 32) # Slightly larger button text
        
        # Menu state
        self.state = "main"  # main, single_agent_type
        
        # Agent types
        self.agent_types = ['RL', 'MPC', 'Random', 'Human']
        self.selected_agent_type = None
        
        # Load configurations
        self.config = self._load_config()
        self.single_agent_config = self._load_single_agent_config()
        
        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        self.fps = 60
        
        # Button definitions
        self.buttons = self._create_buttons()
        
    def _load_background_image(self):
        """Load and scale the background image to match the menu dimensions."""
        try:
            image_path = os.path.join('images', 'main_menu_background.png')
            if os.path.exists(image_path):
                # Load the image
                original_image = pygame.image.load(image_path).convert_alpha()
                # Scale the image to match the menu dimensions
                self.background_image = pygame.transform.scale(original_image, (self.width, self.height))
                print(f"Background image loaded successfully: {image_path}")
            else:
                print(f"Background image not found: {image_path}")
        except Exception as e:
            print(f"Error loading background image: {e}")
    
    def _load_config(self):
        """Load the main configuration file."""
        try:
            with open('config.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config.yaml: {e}")
            return {}
            
    def _load_single_agent_config(self):
        """Load the single agent configuration file."""
        try:
            with open('config_single_agent.yaml', 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config_single_agent.yaml: {e}")
            return {}
    
    def _create_buttons(self):
        """Create button definitions for the menu."""
        button_width = 250 # Slightly wider buttons
        button_height = 55
        center_x = self.width // 2 - button_width // 2
        
        buttons = {
            "main": [
                # Adjusted Y positions and width
                {"text": "Simulation", "rect": pygame.Rect(center_x, 280, button_width, button_height), "action": "simulation"},
                {"text": "Training", "rect": pygame.Rect(center_x, 360, button_width, button_height), "action": "training"},
                {"text": "Exit", "rect": pygame.Rect(center_x, 440, button_width, button_height), "action": "exit"}
            ],
            "simulation": [
                {"text": "Single Agent", "rect": pygame.Rect(center_x, 280, button_width, button_height), "action": "single_agent"},
                {"text": "Multi Agent", "rect": pygame.Rect(center_x, 360, button_width, button_height), "action": "multi_agent"},
                {"text": "Back", "rect": pygame.Rect(center_x, 440, button_width, button_height), "action": "main"}
            ],
            "training": [
                # Adjusted width for longer text
                {"text": "Single Agent Training", "rect": pygame.Rect(self.width // 2 - 300 // 2, 280, 300, button_height), "action": "single_agent_training"},
                {"text": "Multi Agent Training", "rect": pygame.Rect(self.width // 2 - 300 // 2, 360, 300, button_height), "action": "multi_agent_training"},
                {"text": "Back", "rect": pygame.Rect(center_x, 440, button_width, button_height), "action": "main"}
            ],
            "single_agent_type": [
                # Spaced out vertically more
                {"text": "RL", "rect": pygame.Rect(center_x, 220, button_width, button_height), "action": "rl"},
                {"text": "MPC", "rect": pygame.Rect(center_x, 290, button_width, button_height), "action": "mpc"},
                {"text": "Random", "rect": pygame.Rect(center_x, 360, button_width, button_height), "action": "random"},
                {"text": "Human", "rect": pygame.Rect(center_x, 430, button_width, button_height), "action": "human"},
                {"text": "Back", "rect": pygame.Rect(center_x, 500, button_width, button_height), "action": "simulation"}
            ]
        }
        return buttons
    
    def _draw_button(self, button, hover=False):
        """Draw a button with a semi-transparent background."""
        color_rgba = self.button_hover_color if hover else self.button_color
        border_color_rgba = self.button_border_color
        
        # Create a temporary surface for the button with per-pixel alpha
        button_surface = pygame.Surface(button["rect"].size, pygame.SRCALPHA)
        
        # Draw the semi-transparent background onto the temporary surface
        pygame.draw.rect(button_surface, color_rgba, button_surface.get_rect())
        
        # Draw the border onto the temporary surface
        pygame.draw.rect(button_surface, border_color_rgba, button_surface.get_rect(), 2)
        
        # Render the text
        text_surface = self.button_font.render(button["text"], True, self.button_text_color)
        text_rect = text_surface.get_rect(center=button_surface.get_rect().center)
        
        # Blit the text onto the temporary surface
        button_surface.blit(text_surface, text_rect)
        
        # Blit the temporary button surface onto the main screen
        self.screen.blit(button_surface, button["rect"].topleft)
    
    def _draw_title(self, title):
        """Draw the title of the current screen."""
        text_surface = self.title_font.render(title, True, self.text_color)
        # Add a simple shadow effect
        shadow_surface = self.title_font.render(title, True, (30, 30, 30)) 
        shadow_pos = (self.width // 2 + 2, 100 + 2)
        text_pos = (self.width // 2, 100)
        
        shadow_rect = shadow_surface.get_rect(center=shadow_pos)
        text_rect = text_surface.get_rect(center=text_pos)
        
        self.screen.blit(shadow_surface, shadow_rect)
        self.screen.blit(text_surface, text_rect)
    
    def _draw_section(self, title, y_pos):
        """Draw a section title."""
        text_surface = self.section_font.render(title, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.width // 2, y_pos))
        self.screen.blit(text_surface, text_rect)
    
    def _handle_events(self):
        """Handle Pygame events."""
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    for button in self.buttons[self.state]:
                        if button["rect"].collidepoint(mouse_pos):
                            action = button["action"]
                            
                            if action == "single_agent":
                                self.state = "single_agent_type"
                                return True, None
                            elif action in ["rl", "mpc", "random", "human"]:
                                self.selected_agent_type = action.lower()
                                return True, {"mode": "single", "agent_type": self.selected_agent_type}
                            elif action == "multi_agent":
                                return True, {"mode": "multi"}
                            elif action == "single_agent_training":
                                return True, {"mode": "single", "training": True}
                            elif action == "multi_agent_training":
                                return True, {"mode": "multi", "training": True}
                            elif action == "exit":
                                print("Exit button clicked. Closing application.")
                                return False, None
                            elif action in ["main", "simulation", "training"]:
                                self.state = action
                                self.selected_agent_type = None
                                return True, None
        
        return True, None
    
    def _draw(self):
        """Draw the current menu state."""
        # Clear the screen
        self.screen.fill(self.background_color)
        
        # Draw background image if available
        if self.background_image:
            self.screen.blit(self.background_image, (0, 0))
        
        # Draw title based on state
        if self.state == "main":
            self._draw_title("Racing Simulation")
        elif self.state == "simulation":
            self._draw_title("Simulation")
        elif self.state == "training":
            self._draw_title("Training")
        elif self.state == "single_agent_type":
            self._draw_title("Select Agent Type")
        
        # Draw buttons for current state
        mouse_pos = pygame.mouse.get_pos()
        for button in self.buttons[self.state]:
            hover = button["rect"].collidepoint(mouse_pos)
            self._draw_button(button, hover)
        
        # Update the display
        pygame.display.flip()
    
    def run(self):
        """Run the main menu and return the selected options."""
        running = True
        result = None
        
        while running:
            running, new_result = self._handle_events()
            if new_result is not None:
                result = new_result
            self._draw()
            self.clock.tick(self.fps)
            
            # Check if we have a result to return
            if result is not None:
                return result
        
        return None
    
    def close(self):
        """Close the menu and clean up."""
        pygame.quit()
