## Day Ahead Battery Arbitrage Formulation

### Data
Let
- $T$ denote time indices for the optimisation horizon
  - where $\Delta_t$ represents the time step (as a fraction of an hour) between each $t \in T$ (e.g. $\Delta_t = 1$ 
  - for hourly data and $\Delta_t = 0.25$ for quarter-hourly))
  - note: represented as a subset of $\mathbb{Z}$ here for simplicity but can be generalised to any time index set
- $p_t \in \mathbb{R}$ denote the price of electricity (in MWh) at time $t \in T$
- $B \in \mathbb{R}^{>0}$ denotes the battery capacity (in MWh)
- $P^{max} \in \mathbb{R}^{>0}$ denote the maximum (dis)charging power (in MW)
- $\eta_c \in (0,1]$ denote the charging efficiency of the battery
- $\eta_d \in (0,1]$ denote the discharging efficiency of the battery
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

$$ s_t = s_{t-1} + \eta_c \cdot c_t - \frac{d_t}{\eta_d} \ \  \forall t \in T$$
note: defined w.r.t. the grid-side state of charge, i.e. the energy the battery contributes to the grid after accounting for charging and discharging efficiency

##### 2: Do not let state of charge exceed battery capacity

$$s_t \leq B\ \ \forall t \in T$$

##### 3: Do not exceed battery power with charging decision

$$c_t + d_t \leq P^{max}\cdot\Delta_t\ \  \forall t \in T$$

##### 4: Enforce terminal state of charge condition

$$s_{|T|} + c_{|T|} + d_{|T|} = s_0$$


Add later
- battery degradation
- add some terminal storage condition (maybe valued by the present value of the energy and adding to obj. fn.)