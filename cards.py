"""
Provide a random card to the dealer.
"""

import random

def draw_card():
    """
    Return random number between 2 and 11 representing a card value
    """
    return random.randint(2, 11)



