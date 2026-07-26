# Deep Learning from Scratch

Deep learning architectures implemented from scratch in NumPy, applied to real biological data, and validated against PyTorch.

Built as a continuation of my [`machine-learning-from-scratch`](https://github.com/jenniferestigene/machine-learning-from-scratch) repository, moving from classical ML into neural network architectures.

## Approach

Every notebook follows the same structure: implement the architecture's forward pass, loss function, and backpropagation directly in NumPy (no `torch.nn`, no autograd), train it on real data, then validate the result against an equivalent PyTorch implementation with matching architecture and hyperparameters to confirm correctness.

Each architecture here uses the real dataset best suited to its data type: tabular, image, or sequence.

## Results

| Architecture | Task | Dataset | Result | vs. PyTorch |
|---|---|---|---|---|
| [Multi-Layer Perceptron](multi_layer_perceptron.ipynb) | Malignant/benign classification | Breast Cancer Wisconsin (30 real tumor measurements) | 94.69% test accuracy | Close (95.58%) |
| [Convolutional Neural Network](convolutional_neural_network.ipynb) | Pneumonia detection | PneumoniaMNIST (real pediatric chest X-rays) | 83.33% test accuracy | Close (74.36%) |
| [Recurrent Neural Network](recurrent_neural_network.ipynb) | Promoter region classification | Human Non-TATA Promoters (real 251bp genomic sequences) | 71.83% test accuracy | Close (72.67%) |

Where results are "close" rather than exact, it's expected: neural network training is non-convex, so two independent implementations with matching architecture and hyperparameters can converge to different, comparably good solutions depending on random initialization.

## Running locally

```bash
git clone https://github.com/jenniferestigene/deep-learning-from-scratch.git
cd deep-learning-from-scratch
pip install -r requirements.txt
jupyter notebook
```

Unlike the classical ML repo, this one does download real data on first run — MedMNIST and genomic_benchmarks fetch their datasets automatically the first time each notebook loads them.

## License

MIT — see [LICENSE](LICENSE).