# DRSA

This package was designed to manipulate data with DRSA (Dominance-Based Rough Set Approach) usage. Still in development.

### Rule Induction

Implemented the following algorithms:

- [DomLEM](https://link.springer.com/chapter/10.1007/3-540-45554-X_37) algorithm (DRSA),
- [VC-DomLEM](https://www.sciencedirect.com/science/article/abs/pii/S0020025510005359) algorithm with $\epsilon$-consistency (VC-DRSA)
- [extended VC-DomLEM](https://www.sciencedirect.com/science/article/pii/S0377221723007440) algorithm (VC-DRSA with Naive Bayes)

### Classification methods

Supported two classification methods - "standard", based on the intersection of supported classes, and "advanced", based on the score ([article](https://www.sciencedirect.com/science/article/pii/S0377221706001391))

### Other remarks

Class approximations can be generated with thresholds from [this article](https://www.sciencedirect.com/science/article/abs/pii/S0020025510005359):

- $\mu$ (gain type measure)
- $\epsilon^*$ (cost type measure)
