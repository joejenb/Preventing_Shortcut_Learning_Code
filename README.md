# Preventing Shortcut Learning in Convolutional Neural Networks Through Enforced Representation Hierarchy
## Overview 
Shortcut learning can be defined as the use of use of simplistic, low-level characteristics
of stimuli to inform high-level, abstract problems. It is known that representations for such low-level characteristics, form in the initial layers of network's.
Hence this is equivalent to saying that networks that rely on shortcut solutions contain excessive amounts of linearity. With representations formed in early layers propagating to the penultimate layer of the network.
To prevent this we therefore enforce dissimilarity between representations found in the penultimate layer of network and those found in all previous layers. This ultimately
leads to a 15% increase in out of distribution performance for a variant of Coloured-MNIST when used with a ResNet-18 architecture. In addition to this we also make use of BYOL, which is naturally resilient to the shortcut solutions promoted by Coloured-MNIST. Allowing us to asses whether our solution leads to the learning of representations similar to those found in BYOL and what impact it has on already robust representations. We measure intrinsic dimensionality and inter layer similarities throughout training to facilitate this comparison.


![Screenshot from 2022-09-07 22-12-38](https://user-images.githubusercontent.com/30124151/188981545-5b29b16c-945b-4305-b7f6-14dff4078908.png)
![Screenshot from 2022-09-07 22-13-24](https://user-images.githubusercontent.com/30124151/188981591-094b2f35-ff46-40dd-9cf7-734c1869637d.png)
