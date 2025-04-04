# Learning Broken Symmetries with Approximate Invariance

**Seth Nabat, Aishik Ghosh, Edmund Witkowski, Gregor Kasieczka, Daniel Whiteson**

https://doi.org/10.1103/PhysRevD.111.072002

**Abstract:** Recognizing symmetries in data allows for significant boosts in neural network training, which is
especially important where training data are limited. In many cases, however, the exact underlying symmetry is present
only in an idealized dataset, and is broken in actual data, due to asymmetries in the detector, or varying response
resolution as a function of particle momentum. Standard approaches, such as data augmentation or equivariant networks
fail to represent the nature of the full, broken symmetry, effectively overconstraining the response of the neural
network. We propose a learning model which balances the generality and asymptotic performance of unconstrained networks
with the rapid learning of constrained networks. This is achieved through a dual-subnet structure, where one network is
constrained by the symmetry and the other is not, along with a learned symmetry factor. In a simplified toy example that
demonstrates violation of Lorentz invariance, our model learns as rapidly as symmetry-constrained networks but escapes
its performance limitations.
