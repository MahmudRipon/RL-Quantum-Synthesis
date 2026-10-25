#Code written by Mahmud for learning purpose
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set a fixed seed so the training is reproducible (no more random failures)
torch.manual_seed(42)
np.random.seed(42)

# Setup the quantum stuff
I2 = torch.eye(2, dtype=torch.complex64)
X_gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
H_gate = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / np.sqrt(2)
CNOT_gate = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=torch.complex64)


def tp(a, b):
    return torch.kron(a, b)

GATES = [
    tp(H_gate, I2),
    tp(I2, H_gate),
    tp(X_gate, I2),
    CNOT_gate
]
n_actions = len(GATES)

start_state = torch.tensor([1, 0, 0, 0], dtype=torch.complex64)
target_bell = torch.tensor([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=torch.complex64)


# The Neural Net
class AgentBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 64)
        self.lin2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, n_actions)

    def forward(self, state_vec):
        x = torch.cat([state_vec.real, state_vec.imag])
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        return torch.softmax(self.out(x), dim=-1)


def eval_state(current_state, last_score):
    overlap = torch.abs(torch.sum(torch.conj(target_bell) * current_state)) ** 2
    score = overlap.item()

    r = score - last_score
    done = score > 0.99
    if done:
        r += 1.0

    return r, score, done


# Training Loop
def run_training():
    agent = AgentBrain()
    opt = optim.Adam(agent.parameters(), lr=0.01)

    epochs = 600  # Increased epochs slightly
    max_steps_per_ep = 10
    gamma = 0.99

    print("Starting training...")

    for ep in range(epochs):
        state = start_state.clone()
        init_overlap = torch.abs(torch.sum(torch.conj(target_bell) * state)) ** 2
        prev_score = init_overlap.item()

        log_probs_list = []
        rewards_list = []
        entropies = []

        for step in range(max_steps_per_ep):
            probs = agent(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs_list.append(dist.log_prob(action))
            entropies.append(dist.entropy())  # Add entropy for exploration

            gate_mat = GATES[action.item()]
            state = torch.matmul(gate_mat, state)

            reward, score, is_done = eval_state(state, prev_score)
            rewards_list.append(reward)
            prev_score = score

            if is_done:
                break

        returns = []
        R = 0
        for r in reversed(rewards_list):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = []
        for log_prob, R, entropy in zip(log_probs_list, returns, entropies):
            # Entropy bonus (0.05) encourages the agent not to get stuck!
            loss.append(-log_prob * R - 0.05 * entropy)

        opt.zero_grad()
        total_loss = torch.stack(loss).sum()
        total_loss.backward()
        opt.step()

        if (ep + 1) % 50 == 0:
            print(f"Ep {ep + 1} | Best fidelity: {prev_score:.4f}")
            if prev_score > 0.99:
                print("Found the Bell state! Agent figured it out.")
                break


if __name__ == "__main__":
    run_training()