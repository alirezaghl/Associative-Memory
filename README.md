# Associative Memory Architectures

Implementation of associative memory architectures including the original Hopfield networks and modern versions of them, so-called continuous modern Hopfield networks. These models serve as content-addressable memory units. The units in a Hopfield network are binary threshold units; in the original design, every unit would take either 1 or -1. Between each unit there is a symmetric connection, and these connections determine the tendency to set the unit's state. There is a threshold for each unit, and the state of a unit depends on whether the weighted sum of inputs from other units exceeds this threshold.

There is no self-recurrence in units. These connections are learned via the Hebbian learning rule. The network dynamics are asynchronous, where at each time step, only one unit gets updated. Once the network has learned the associations, there would not be any weight changes until new patterns are introduced to the network.

These architectures are important because they learn associations and can serve as biologically plausible networks (kind of), and there is no need for updating the parameters of the network with gradients and backpropagation.


## References

- [Neural networks and physical systems with emergent collective computational abilities](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/)
- [Hopfield Networks is All You Need](https://ml-jku.github.io/hopfield-layers/)
- [Walkthrough of "Hopfield Networks are all you need"](https://www.beren.io/2020-11-02-Walkthrough_Hopfield-Networks-Is-All-You-Need/)
