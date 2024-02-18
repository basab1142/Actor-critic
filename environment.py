import numpy as np
class CATCHTHEBALL:
    def __init__(self, N: int):
        self.size = N
        self.frame = np.zeros(shape = (N, N), dtype = np.int8)
        self.moving_down = 1 # If the ball is moving down
        self.deviation_left = 1 # whether balls moving in left or right 1 -> left deviation
        self.ball_position = [ 1 , np.random.randint(N//4, N//2)]
        self.plate_position = [ N-2 , np.random.randint(1, N - 2)]
    def ballMove(self):
        N = self.size
        # if ball is by the wall then the deviatin changes and move according to
        if tuple(self.ball_position) == tuple([self.plate_position[0] -1 ,self.plate_position[1]]): # just above the plate
            if self.ball_position[1] == 1 or self.ball_position[1] == N - 2: # edge
                self.moving_down = -1 # going up
            else:
                self.moving_down = - 1
                self.deviation_left = -1*self.deviation_left
        elif self.ball_position[0] == 1 and (self.ball_position[1] == 1 or self.ball_position[1] == N-2) and self.moving_down==-1:# exaxt edge
            if self.ball_position[1] == 1:               
                self.deviation_left = -1
            else:
                self.deviation_left = 1
            self.moving_down = 1
                
        elif self.ball_position[1] == 1 or self.ball_position[1] == N-2: #
            self.deviation_left = -1*self.deviation_left
        
        elif self.ball_position[0] == 1 and self.moving_down == -1:# going up and edge
            if self.ball_position[1] == 1 or self.ball_position[1] == N - 2: # edge
                self.moving_down = 1 # going up
            else:
                self.moving_down =  1
                self.deviation_left = -1*self.deviation_left
            
        
        # move now
        self.ball_position[0] += self.moving_down
        self.ball_position[1] -= self.deviation_left*self.moving_down
        
    
    def movePlate(self, action : int): # 0-> left 1->right 2-> nothing
        if action == 0: # move left
            if self.plate_position[1] == 1: # in the edge and move left
                self.plate_position[1] += 1
            else:
                
                self.plate_position[1] -= 1    
        elif action == 1:
            if self.plate_position[1] == self.size - 2:
                self.plate_position[1] -= 1
            else:
                
                self.plate_position[1] += 1    
    
    def getFeature(self): # [relative position, going_up, left_directed], reward, done
        
        pos = [self.ball_position[0] - self.plate_position[0], self.ball_position[1] - self.plate_position[1], 
              self.moving_down, self.deviation_left]
        if self.ball_position[0] == self.size - 2:
            return pos, 0, 1
        elif tuple(self.ball_position) == tuple([self.plate_position[0] -1 ,self.plate_position[1]]):
            return pos, 1, 0
        else:
            return pos, 0, 0
                                                
        
    def showFrame(self):
        # present ball as `o` and `_` plate
        for i in range(self.size):
            for j in range(self.size):
                if i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                    print('*', end = ' ')
                elif tuple(self.ball_position) == (i, j):
                    print('o', end = ' ')
                elif tuple(self.plate_position) == (i, j):
                    print('-', end = ' ')
                else:
                    print(' ', end = ' ')
            print()        