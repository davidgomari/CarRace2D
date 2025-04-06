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
        
        # Colors
        self.background_color = (50, 50, 50)
        self.text_color = (255, 255, 255)
        self.button_color = (70, 70, 70)
        self.button_hover_color = (100, 100, 100)
        self.button_text_color = (255, 255, 255)
        self.section_color = (60, 60, 60)
        
        # Load background image
        self.background_image = None
        self._load_background_image()
        
        # Fonts
        self.title_font = pygame.font.SysFont(None, 48)
        self.section_font = pygame.font.SysFont(None, 36)
        self.button_font = pygame.font.SysFont(None, 28)
        
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
        buttons = {
            "main": [
                {"text": "Simulation", "rect": pygame.Rect(300, 200, 200, 50), "action": "simulation"},
                {"text": "Training", "rect": pygame.Rect(300, 300, 200, 50), "action": "training"},
                {"text": "Exit", "rect": pygame.Rect(300, 400, 200, 50), "action": "exit"}
            ],
            "simulation": [
                {"text": "Single Agent", "rect": pygame.Rect(300, 200, 200, 50), "action": "single_agent"},
                {"text": "Multi Agent", "rect": pygame.Rect(300, 300, 200, 50), "action": "multi_agent"},
                {"text": "Back", "rect": pygame.Rect(300, 400, 200, 50), "action": "main"}
            ],
            "training": [
                {"text": "Single Agent Training", "rect": pygame.Rect(250, 200, 300, 50), "action": "single_agent_training"},
                {"text": "Multi Agent Training", "rect": pygame.Rect(250, 300, 300, 50), "action": "multi_agent_training"},
                {"text": "Back", "rect": pygame.Rect(300, 400, 200, 50), "action": "main"}
            ],
            "single_agent_type": [
                {"text": "RL", "rect": pygame.Rect(300, 200, 200, 50), "action": "rl"},
                {"text": "MPC", "rect": pygame.Rect(300, 270, 200, 50), "action": "mpc"},
                {"text": "Random", "rect": pygame.Rect(300, 340, 200, 50), "action": "random"},
                {"text": "Human", "rect": pygame.Rect(300, 410, 200, 50), "action": "human"},
                {"text": "Back", "rect": pygame.Rect(300, 480, 200, 50), "action": "simulation"}
            ]
        }
        return buttons
    
    def _draw_button(self, button, hover=False):
        """Draw a button on the screen."""
        color = self.button_hover_color if hover else self.button_color
        pygame.draw.rect(self.screen, color, button["rect"])
        pygame.draw.rect(self.screen, self.text_color, button["rect"], 2)
        
        text_surface = self.button_font.render(button["text"], True, self.button_text_color)
        text_rect = text_surface.get_rect(center=button["rect"].center)
        self.screen.blit(text_surface, text_rect)
    
    def _draw_title(self, title):
        """Draw the title of the current screen."""
        text_surface = self.title_font.render(title, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.width // 2, 100))
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
