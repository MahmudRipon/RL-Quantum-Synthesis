# RL for Quantum Circuit Synthesis (Learning Project)

I put this small project together on my laptop because I wanted to bridge my math/physics background with something practical in AI. I was reading the recent *Phys. Rev. X* paper on using Reinforcement Learning to discover fault-tolerant quantum circuits, and I figured the best way to truly understand it was to build a tiny version myself. 

This is a PyTorch implementation of a REINFORCE agent that learns how to build a Bell State (`|00> + |11>)/√2`) from scratch, starting at `|00>`. The agent gets zero hints—it just applies quantum gates, looks at the resulting state vector, and gets a reward based on how close it is to the target.

### A classic RL roadblock
When I first ran it, the agent kept getting stuck at exactly `0.5000` fidelity. It had figured out that applying a Hadamard gate creates a superposition, but it couldn't take the next step to apply the CNOT gate and actually entangle the qubits. It was a textbook local minimum.

To fix this, I added an **entropy bonus** to the loss function. This penalizes the agent for becoming too confident in one specific action too early, forcing it to keep exploring the other gates. I also set a manual seed (`torch.manual_seed(42)`) so my runs were reproducible and I could actually debug it. Once I fixed the exploration, it found the solution in about 100 episodes.

### How it works under the hood
- **The Environment:** A simple 2-qubit simulator using PyTorch complex tensors.
- **Actions:** H on q0, H on q1, X on q0, or CNOT.
- **The Brain:** A basic 3-layer MLP that takes the flattened real/imaginary parts of the state vector and outputs probabilities for the 4 gates.
- **Reward:** The agent gets a reward every time the fidelity increases. If it hits >0.99, it gets a big bonus and the episode ends.

### What it looks like when it works
```text
Starting training...
Ep 50 | Best fidelity: 0.5000
Ep 100 | Best fidelity: 1.0000
Found the Bell state! Agent figured it out.
