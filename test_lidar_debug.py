#!/usr/bin/env python3
"""
Debug script to test LiDAR calculation and verify it's working correctly.
"""

import yaml
import numpy as np
from environment import RacingEnv

def test_lidar_calculation():
    """Test the LiDAR calculation with a simple scenario."""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable LiDAR printing for debugging
    config['simulation']['print_lidar'] = True
    config['simulation']['num_episodes'] = 1
    config['simulation']['max_steps'] = 10  # Just a few steps for testing
    
    # Create environment
    env = RacingEnv(config=config, mode='multi')
    
    print("=== LiDAR Debug Test ===")
    print(f"Track dimensions: length={env.track.length}, radius={env.track.radius}, width={env.track.width}")
    print(f"LiDAR config: beams={env.lidar_num_beams}, max_range={env.lidar_max_range}")
    print()
    
    # Reset environment
    observations, infos = env.reset()
    
    # Test LiDAR calculation for each agent
    for agent_id, car in env.cars.items():
        print(f"Agent {agent_id} position: x={car.x:.2f}, y={car.y:.2f}, theta={car.theta:.2f}")
        
        # Calculate LiDAR manually
        all_car_data = {aid: c.get_data() for aid, c in env.cars.items()}
        lidar_data = env._calculate_lidar(agent_id, all_car_data)
        
        print(f"LiDAR data: {lidar_data.cpu().numpy()}")
        print(f"Min distance: {np.min(lidar_data.cpu().numpy()):.2f}")
        print(f"Max distance: {np.max(lidar_data.cpu().numpy()):.2f}")
        print(f"Mean distance: {np.mean(lidar_data.cpu().numpy()):.2f}")
        print()
    
    # Test a few steps
    print("=== Running a few simulation steps ===")
    for step in range(5):
        # Get random actions
        actions = {}
        for agent_id in env.agents.keys():
            actions[agent_id] = np.array([0.5, 0.0], dtype=np.float32)  # Moderate throttle, no steering
        
        observations, rewards, terminated, truncated, infos = env.step(actions)
        
        print(f"Step {step + 1}:")
        for agent_id, obs in observations.items():
            if 'lidar' in obs:
                lidar = obs['lidar'].cpu().numpy()
                print(f"  {agent_id}: min={np.min(lidar):.2f}, max={np.max(lidar):.2f}, mean={np.mean(lidar):.2f}")
        print()
    
    env.close()
    print("LiDAR debug test completed.")

def test_track_boundaries():
    """Test track boundary detection."""
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    env = RacingEnv(config=config, mode='multi')
    
    print("=== Track Boundary Test ===")
    
    # Test points that should be on track
    test_points_on_track = [
        (0, -env.track.radius),  # Center of bottom straight
        (0, env.track.radius),   # Center of top straight
        (env.track.length/2, 0), # Right curve center
        (-env.track.length/2, 0), # Left curve center
    ]
    
    # Test points that should be off track
    test_points_off_track = [
        (0, -env.track.outer_radius - 1),  # Too far outside
        (0, env.track.outer_radius + 1),   # Too far outside
        (env.track.length/2 + 1, 0),       # Too far right
        (-env.track.length/2 - 1, 0),      # Too far left
    ]
    
    print("Points that should be ON track:")
    for x, y in test_points_on_track:
        on_track = env.track.is_on_track(x, y)
        print(f"  ({x:.1f}, {y:.1f}): {on_track}")
    
    print("Points that should be OFF track:")
    for x, y in test_points_off_track:
        on_track = env.track.is_on_track(x, y)
        print(f"  ({x:.1f}, {y:.1f}): {on_track}")
    
    env.close()

if __name__ == "__main__":
    print("LiDAR Debug Script")
    print("=" * 40)
    print()
    
    test_track_boundaries()
    print()
    test_lidar_calculation()
