# DRSA

## What this package is about?

This package was designed to manipulate data with DRSA (Dominance-Based Rough Set Approach) usage.

To induce rules, there'll be implemented:

- [DomLEM](https://link.springer.com/chapter/10.1007/3-540-45554-X_37) algorithm (DRSA),
- [VC-DomLEM](https://www.sciencedirect.com/science/article/abs/pii/S0020025510005359) algorithm with $\epsilon$-consistency (VC-DRSA)
- [extended VC-DomLEM](https://www.sciencedirect.com/science/article/pii/S0377221723007440) algorithm (VC-DRSA with Naive Bayes)

Other remarks:

- class approximations can be generated with thresholds from [this article](https://www.sciencedirect.com/science/article/abs/pii/S0020025510005359):
  - $\mu$ (gain type measure)
  - $\epsilon^*$ (cost type measure)
