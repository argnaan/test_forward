import argparse
import subprocess

parser = argparse.ArgumentParser("forward on Gap8")
parser.add_argument("--dim", type=int, default=64)
parser.add_argument("--hidden_dim", type=int, default=172)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--n_kv_heads", type=int, default=4)
parser.add_argument("--vocab_size", type=int, default=512)
parser.add_argument("--seq_len", type=int, default=512)
parser.add_argument("--steps", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--rnd_seed", type=int, default=42)
args = parser.parse_args()

dim = args.dim
hidden_dim = args.hidden_dim
n_layers = args.n_layers
n_heads = args.n_heads
n_kv_heads = args.n_kv_heads
vocab_size = args.vocab_size
seq_len = args.seq_len
steps = args.steps
temperature = args.temperature
rnd_seed = args.rnd_seed

subprocess.run(["utils/genRndWeights", str(dim), str(hidden_dim), str(n_layers), str(n_heads), str(n_kv_heads), str(vocab_size), str(seq_len), str(steps), str(temperature), str(rnd_seed)])
subprocess.run(["utils/run", "utils/rnd_weights.bin", "-z", "utils/tokenizer.bin", "-t", str(temperature), "-s", str(rnd_seed)])