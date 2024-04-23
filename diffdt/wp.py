import torch as th

'''
Weighted points.
'''
class WPoints:
    def __init__(self, 
                positions: th.Tensor, 
                weights: th.Tensor):
        '''
        @ position: [# point, # dim], position of the point
        @ weight: [# point], weight of the point
        '''
        self.positions = positions
        self.weights = weights