# Preventing Shortcut Learning in Convolutional Neural Networks Through Enforced Representation Hierarchy
## Overview 
Shortcut learning can be defined as the use of use of simplistic, low-level characteristics
of stimuli to inform high-level, abstract problems. It is known that representations for such low-level characterics, form in the initial layers of network's.
Hence this is equivalent to saying that networks that rely on shortcut solutions contain excessive amounts of linearity. With representations formed in early layers propagating to the penultimate layer of the network.
To prevent this we therefore enforce dissimilarity between representations found in the penultimate layer of network and those found in all previous layers. This ultimately
leads to a 15% increase on a variant of Coloured-MNIST when used with a ResNet-18 architecture. In addition to this we also look at the impact of these methods when combined ...
