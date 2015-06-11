Space Warps Cutouts
-------------------

Crowd-sourced discovery of gravitational lenses. We are a Zooniverse project,
which launched in Spring 2013 at
[http://spacewarps.org](http://spacewarps.org). This tar contains cutouts of
lens candidates and lens-like false positives selected by the crowd, an
associated catalog, an IPython Notebook to help get you started, an associated
python script containing some 'helper' functions, and, of course, this
README.md

Two papers on the first dataset we searched for lenses, the CFHTLS, can be found here:

* [Marshall et al 2015: Crowd-sourcing the Discovery of Gravitational Lenses](http://arXiv.org/abs/1504.06148)
* [More et al 2015: Results from CFHTLS](http://arxiv.org/abs/1504.05587)

The web app itself is being developed
[here](https://github.com/zooniverse/Lens-Zoo) as part of the
[Zooniverse](http://zooniverse.org) project. If you are interested in building
a site like Space Warps using Zooniverse techology, keep an eye on their
website or drop them a line.


#### Brief note on nomenclature

Each row in this catalog is an 'object' that is
derived from the 'fields' that SpaceWarps volunteers examined. There can be
multiple objects per field, and their flavors may disagree. For example, a
field may have a simulated lensing cluster, but some objects may be of regions
that are not that lens, but instead some kind of dud. Users may also mark
multiple objects in a given field.


### Description of Object Identification Algorithm

For a given field, a user may mark zero or multiple locations where they
believe a strong gravitational lens resides. We take all training fields in
stage 1 and 2 of as well as the test fields from stage 2. Objects are
identified from the markers users place. (So if a field is entirely unmarked,
then no objects will be identified in it.) Using these markers, we identify
clusters using the Density-based spatial clustering of applications with noise
(DBSCAN) algorithm as implemented in scikit learn. Briefly: the algorithm takes
points and looks for points in some neighborhood about the point. If there are
enough points, then a cluster is identified and membership is grown
agglomeratively until there are no more points within some distance of any
cluster members. As of the latest version, the cluster identification is
entirely deterministic as done by scikit learn. The DBSCAN algorithm has two
hyperparameters: the minimum number of markers, and the neighborhood distance.
For this specific implementation we use a neighborhood of thirty pixels and
require at least three markers to form a cluster. If a user places multiple
markers on a field, then each point is downweighted by that number, e.g. if a
user places two markers, each marker is now only weighted one half a marker
when the DBSCAN algorithm is run.

### Catalog Description

ID : SpaceWarps Field ID
ZooID : Zooniverse Field ID
location : Online location of the field image.
mean_probability : Probability as derived by the SpaceWarps system.
category : [training, test] Training objects have a known truth value, while test do not.
kind : [dud, sim, test] Duds and sims are subcategories of the training objects indicating the negative or positive presence of a simulated lens in the field.
field_flavor : [dud, test, lensing cluster, lensed galaxy, lensed quasar] Gives the specific type of simulated field.
state : [active, inactive] At the end of the SpaceWarps survey, was this object still in the system?
status : [rejected, detected, undecided] Was the mean_probability below 1e-7, above 0.95, or in the middle?
truth : [NOT LENS UNKNOWN] Just like kind.
stage : [1, 2] SpaceWarps was conducted in two stages. The first had easier training images and sought to filter out many dudes, while the second meant to refine the remaining to find a good candidate list.
field_name : Filename of the field image.
object_name : Filename of the object image.
object_flavor : ['dud' 'simulated lensing cluster' 'simulated lensed galaxy'
 'simulated lensed quasar' 'known lens' 'unknown'] The flavor of the object. This is the column you would use to train your network!
object_id : Unique number associated with each object.
x : Object x centroid in pixel coordinates of the field.
y : Object y centroid in pixel coordinates of the field.
object_size : Standard deviation of marker coordinates with respect to the centroid.
skill_sum : Each user has a skill as assessed by SpaceWarps. This is represents the total sum of that skill in a cluster.
markers_in_object : Number of markers associated with the object.
markers_in_field : Total number of markers in the field.
people_marking_objects : Total number of people who marked the object.
people_marking_field : Total number of people who marked the field.
people_looking_at_field : Total number of people who looked at the field (and may or may not have marked it).

### Contact

* [Chris Davis](cpd@stanford.edu) (KIPAC, SLAC National Accelerator Laboratory)
* [Phil Marshall](http://drphilmarshall.net) (KIPAC, SLAC National Accelerator Laboratory)
* Aprajita Verma (Physics Department, University of Oxford)
* Anupreeta More (Kavli IPMU, University of Tokyo)

spacewarpspi@googlegroups.com

### License

All our code is free to re-use under the GPL v2 license, which just means you
have to make yours available in the same way. If you make use of any of it in
your research, please cite us at this website, and (for now) as *"(Marshall et
al, More et al in preparation)"*. Please do get in touch though - it would be
great to collaborate on improving the SWAP analysis, for example!
