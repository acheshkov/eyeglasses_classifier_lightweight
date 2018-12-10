from typing import List, Generator, Any, Dict, Tuple, Type
import torch
import math

class StopCriteria(object):
    ''' Class helps find the point when we need stop training process. 
    It stops when neither score nor loss improved '''
    
    def __init__(self,  no_improve_th: int = 4):
        self.best_loss = math.inf
        self.best_score = -math.inf
        self.best_model_params = None
        self.no_improve_counter = 0
        self.no_improve_th = no_improve_th

    def check(self, loss: float, score: float, model: torch.nn.Module) -> bool:
        if (score > self.best_score):
            self.no_improve_counter = 0
            if (score  > self.best_score): self.save_best_model_params(model)
            self.best_score = score
            self.best_loss = min(self.best_loss, loss)
        elif loss < self.best_loss:
            self.best_loss = loss
        else:  # neither loss nor score improved 
            self.no_improve_counter +=1 
        
        return self.no_improve_counter >= self.no_improve_th

    def get_best_model_params(self):
        return self.best_model_params

    def save_best_model_params(self, model):
        self.best_model_params = model.state_dict()
        torch.save(model.state_dict(), './best_model_params')
        

    