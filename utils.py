# utils.py
import numpy as np

def check_collision(car1, car2):
    """ Simple circle-circle collision check. """
    dist_sq = (car1.x - car2.x)**2 + (car1.y - car2.y)**2
    min_dist_sq = (car1.collision_radius + car2.collision_radius)**2
    return dist_sq < min_dist_sq