# pic2ohms
Get the value of a resistor from a picture.

# Summary

pic2ohms works in three stages: first a neural network scans the input picture and localizes resistors. These resistors are passed to a second neural network that tries to detect the orientation they are in. Finally a group of three neural networks identify the colors of the bands on each resistor (so far it ignores the tolerance ring or resistors with four bands).

# Status

Right now the scripts that generate the training data produce images with very little variation, with only a few hundred possible outputs, so of course the NN overtrain and get 100% accuracy. The next step is then to improve the data generation.
