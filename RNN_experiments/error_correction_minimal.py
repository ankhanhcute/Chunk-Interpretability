
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(1)

# 1. Generate training sequence

sequence = "E"
for _ in range(1000):
    sequence += "ABCD" + "E" * np.random.randint(0, 20)

chars = sorted(list(set(sequence)))
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for ch, i in char_to_int.items()}

input_size = len(chars)
hidden_size = 12
output_size = len(chars)


# 2. Define simple RNN

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden):
        combined = torch.cat((input_tensor, hidden), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


def one_hot(ch):
    return torch.nn.functional.one_hot(
        torch.tensor([char_to_int[ch]], dtype=torch.long),
        num_classes=input_size
    ).float()

# 3. Train RNN to predict next char

def train_rnn(rnn, sequence, n_iters=500, chunk_len=200, lr=0.005):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(rnn.parameters(), lr=lr)

    for it in range(1, n_iters + 1):
        hidden = rnn.initHidden()
        optimizer.zero_grad()

        start = random.randint(0, len(sequence) - chunk_len - 2)
        seq = sequence[start:start + chunk_len + 1]

        total_loss = 0
        correct = 0
        total = 0

        for t in range(len(seq) - 1):
            x = one_hot(seq[t])
            y = torch.tensor([char_to_int[seq[t + 1]]], dtype=torch.long)

            output, hidden = rnn(x, hidden)
            loss = criterion(output, y)
            total_loss += loss

            pred = output.argmax(1).item()
            correct += int(pred == y.item())
            total += 1

        total_loss.backward()
        optimizer.step()

        if it % 50 == 0 or it == 1:
            print(f"iter {it:4d} | loss {total_loss.item()/total:.4f} | acc {correct/total:.4f}")

    return rnn


# 4. Evaluate corruption behavior

def run_prompt(rnn, prompt, steps=4):
    hidden = rnn.initHidden()
    outputs = []

    for ch in prompt:
        output, hidden = rnn(one_hot(ch), hidden)
        pred = int_to_char[output.argmax(1).item()]
        outputs.append((ch, pred, torch.exp(output).detach().numpy()[0]))

    # continue autoregressively
    current = prompt[-1]
    for _ in range(steps):
        output, hidden = rnn(one_hot(current), hidden)
        pred = int_to_char[output.argmax(1).item()]
        outputs.append((current, pred, torch.exp(output).detach().numpy()[0]))
        current = pred

    return outputs


def print_eval(name, prompt):
    print("\n" + "=" * 60)
    print(name, "| prompt:", prompt)
    print("=" * 60)
    outputs = run_prompt(rnn, prompt)
    for inp, pred, probs in outputs:
        prob_dict = {int_to_char[i]: round(float(p), 3) for i, p in enumerate(probs)}
        print(f"input {inp} -> predicted next {pred} | probs {prob_dict}")


rnn = RNN(input_size, hidden_size, output_size)
rnn = train_rnn(rnn, sequence, n_iters=400)

# Clean learned chunk
print_eval("Clean chunk", "ABCD")

# Error correction / corrupted variants
print_eval("Missing element variant AB?D approximated as ABD", "ABD")
print_eval("Repeated element variant ABBCD", "ABBCD")
print_eval("Early cutoff AB", "AB")
print_eval("Single start A", "A")
