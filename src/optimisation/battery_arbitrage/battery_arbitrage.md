## Day Ahead Battery Arbitrage Formulation

### Data
Let
- $T = \{0, 2, ..., 23\}$ denote time indices at hourly intervals for the following day
- $p_t \in \mathbb{R}^{\ge 0}$ denote the price of electricity (in MWh) at time $t \in T$
- $B \in \mathbb{R}^{>0}$ denotes the battery capacity (in MWh)
- $P \in \mathbb{R}^{>0}$ denote the maximum (dis)charging power (in MW)
- $s_0 \in \mathbb{R}^{\ge 0}$ denote the initial state of charge (in MWh)

### Variables 
Let
- $d_t \in \mathbb{R}^{\ge 0}$ denote the energy discharged (in mWh) during hour $t$ 
- $c_t \in \mathbb{R}^{\ge 0}$ denote the energy charged (in mWh) during hour $t$ 
- $s_t \in \mathbb{R}^{\ge 0}$ denote the state of charge (in mWh) for the battery at time $t$, where $s_0$ is given

### Objective Function

$$\min \sum_{t \in T_d} p_t(d_t-c_t)$$

### Constraints

##### 1: Define state of charge for each device at each time point using the Markov property

$$ s_t = s_{t-1} + c_t - d_t\ \  \forall t \in T$$

##### 2: Reach each device's target state of charge at relevant time points

$$c_{d, t} \geq l_{d, t}\ \ \forall d \in D \ \ \forall t \in \tilde{T}_d$$

##### 3: Do not let state of charge exceed battery capacity

$$s_t \leq B\ \ \forall t \in T$$

##### 4: Do not exceed battery power with charging decision

$$c_t + d_t \leq P\ \  \forall t \in T$$


Add later
- charging and discharging efficiency
- battery degradation
- $\Delta_t$ to generalise time steps from hourly