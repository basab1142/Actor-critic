* `Loss for critic network`: $\delta^2 = (r(s,a) + \gamma*V(S') - V(S))^2$
* `Loss for actor netwrok`: $\delta*ln(\pi(a|s)$) 

## ABOUT Actor-critic

* [Policy-Based] : An Actor that controls how the agent behaves in a situation 
* [Value-Based] : A Critic that measures how good the  action taken by the agent is. 


## Overview of algorithm
1. Initialiaze actor critic algorithm
2. FOR Large number of episodes:
   * Reset Env, score, terminal flag
   * While state is not terminal:
     * Select action acoring to actor network
     * Take action and receive new reward and state
     * calculate $\delta$