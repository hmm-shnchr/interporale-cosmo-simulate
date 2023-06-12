# interporale-cosmo-simulate
Interpolating Cosmological-N-Body-Simulation between times in a Machine-Learning-Model. Use existing MergerTree of simulation data as a dataset for training the Machine-Learning-Model. It is a Multilayer-Neural-Network consisting of all coupled layers.<br>

Hosthalo and subhalos are identified by Rockstar([Behroozi et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...762..109B/abstract)). Consistent tree([Behroozi et al. 2013](http://adsabs.harvard.edu/abs/2013ApJ...763...18B)) is used for MergerTree construction.<br>

Prepared [sample datas](/make-pickle-mergertree/sample-params) in this repository.
These sample datas are used to train Machine-Learning-Model.
A method for making datasets from MergerTree(.tree) is described in [/make-pickle-mergertree/README.md](/make-pickle-mergertree/README.md). and [/make-dataset/README.md](/make-dataset/README.md) <br>

Once the datasets has benn created, run a following command to train Machie-Learning-Model.
`python main.py`