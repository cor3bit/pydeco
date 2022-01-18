<p align="center">
  <img width="215" height="38" src="./artifacts/logo2.png">
</p>

<h2 align='center'>Experiments in Decentralized and Distributed Control Algorithms</h2>

----

### Usage

- Install the library with `pip`:

```
$ pip install git+https://github.com/cor3bit/pydeco.git
```

TODO



### Supported Environments

* Vanilla LQ
* Centralized LQ (each agent has access to the full state)
* Distributed LQ (local information)
* Non-linear Vehicle Platoon Model [in progress]

### Supported Controllers

* Single-agent Analytical LQR
* Single-agent GPI
  * RLS Policy Evaluation
  * Q-learning Policy Evaluation
  * Q-learning GN Policy Evaluation
* Multi-agent GPI
  * RLS Policy Evaluation
  * Q-learning Policy Evaluation
  * Q-learning GN Policy Evaluation


### Acknowledgements

TODO


### References

1. Steven J. Bradtke, B. Erik Ydstie, Andrew G. Barto, 
"Adaptive linear quadratic control using policy iteration" (1994)
2. Siavash Alemzadeh, Mehran Mesbahi, 
"Distributed Q-Learning for Dynamically Decoupled Systems" (2018)
3. Daniel Goerges, "Distributed Adaptive Linear Quadratic Control 
using Distributed Reinforcement Learning" (2019)
4. Hang Wang, Sen Lin, H. Jafarkhani, Junshan Zhang, "Distributed 
Q-Learning with State Tracking for Multi-agent Networked Control" (2020)