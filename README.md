<div align="center">

# A Comparative Analysis of Optimization Trajectories and Convergence Properties

*Geometric analysis of SGD, Adam, Lion and Muon on Fashion-MNIST*

</div>

---

## Overview

<details>
<summary><strong>Abstract</strong></summary>
<br>

The choice of the optimization algorithm determines training efficiency and generalization of deep neural networks. While methods such as Stochastic Gradient Descent and Adam have become field standards, recent works such as **Lion** and **Muon** promise better convergence properties in practice.

In this paper we present a comparative study of the optimization trajectories produced by SGD, Adam, Lion and Muon applied to a Multi-Layer Perceptron on the Fashion-MNIST dataset. To this end, we use a **linear interpolation method**: for parameters before and  after an epoch, we evaluate the loss function at N=50 points along the connecting vector, allowing us to empirically verify convexity and smoothness properties along the trajectory.

Furthermore, we evaluate the convergence points by testing a first-order characterization of local convexity
for points within a neighborhood of convergence point.

</details>

This project is a course paper produced as part of **[CS-439 — Optimization for Machine Learning](https://edu.epfl.ch/coursebook/en/optimization-for-machine-learning-CS-439)** at [EPFL](https://www.epfl.ch). It goes beyond standard aggregate metrics (final loss, test accuracy) to study the *geometry* of each optimizer's path through parameter space: local convexity, empirical smoothness, and whether convergence points satisfy optimality conditions.

---

## Running the experiment

Open and run `main.ipynb` end-to-end.

---

## Authors

| Name | Email | Institution |
|------|-------|-------------|
| Ozair Faizan | [ahmad.faizan@epfl.ch](mailto:ahmad.faizan@epfl.ch) | EPFL |
| Paul Tercier | [paul.tercier@epfl.ch](mailto:paul.tercier@epfl.ch) | EPFL |

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

MIT is the standard open license for academic code releases: it allows anyone to use, modify, and distribute the code freely while preserving attribution.
