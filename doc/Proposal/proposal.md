CS231 Project Proposal

Using Convolutional Neural Networks to Recognize Strong Gravitational Lenses

Chris Davis, Andrew McLeod

A consequence of Einstein's Theory of General Relativity is that mass bends the
path of light. When light passes through a particularly deep gravitational
potential, the deflections, called strong gravitational lenses, can produce
brilliant arcs and multiple images. The number and distribution of strong
lenses reflect the expansion history of the universe.

In this project we will use images collected by the Canada-France-Hawaii
Telescope Legacy Survey to analyze how Convolution Neural Networks can improve
automated detection of strong lens systems. From other graduate work (but not
coursework), we have the locations and categories of around one hundred and
twenty known strong lenses, two hundred large fields verified to contain no
strong lenses, two hundred simulated strong lenses, and several thousand
classifications by citizen-scientists of other potential strong lens systems.
These will form the core of our training and testing datasets; our metric will
be how well a CNN correctly identifies known and simulated lenses and non-lens
systems.

We would like to examine the following questions:
     - Do we have enough data to reasonably train and test a CNN? Can we get
       around this by artificially inflating the data, e.g. by adding rotated
       images?
     - What processing needs to be done on the data? What kind of scaling of
       pixel data is appropriate for automated detection? Should we compress
       the five different 'colors' to a smaller number of dimensions?
     - How do citizen-scientists do compared with this automated system?
     - Can we use the results of citizen-scientists to train the CNN?
     - What sets of classifications are needed? Are we better served sticking
       to 'lens' and 'not lens', or should we use several classification
       categories ('lensed arcs', 'lensed multiple images', 'non-lens pixel
       noise')?
Results will likely be presented in terms of the number of false positives and
false negatives of the different categories of lenses. We intend to compare
these results with other methods in the literature.

An example set of lens and difficult non-lens systems (taken from a manuscript
in preparation) can be found here:
https://www.dropbox.com/s/z8h7g095h7wxm3i/Gallery.png .


TODO: (subject: [cs231n] project proposal + your SUNet Ids) of your project proposal to cs231n-winter1415-staff@lists.stanford.edu.
