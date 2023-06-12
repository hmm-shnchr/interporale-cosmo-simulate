# interporale-cosmo-simulate
Interpolating Cosmological-N-Body-Simulation between times in a Machine-Learning-Model.
Use existing MergerTree of simulation data as a dataset for training the Machine-Learning-Model.
It is a Multilayer-Neural-Network consisting of all coupled layers.<br>

Hosthalo and subhalos are identified by Rockstar([Behroozi et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...762..109B/abstract)).
Consistent tree([Behroozi et al. 2013](http://adsabs.harvard.edu/abs/2013ApJ...763...18B)) is used for MergerTree construction.<br>

Prepared [sample datas](/make-pickle-mergertree/sample-params) in this repository.
These sample datas are used to train Machine-Learning-Model.
A method for making datasets from MergerTree(.tree) is described in [/make-pickle-mergertree/README.md](/make-pickle-mergertree/README.md) and [/make-dataset/README.md](/make-dataset/README.md).<br>

Once the datasets has been created, rewrite [/mylib/learning_parameter.py](/mylib/learning_parameter.py), which defines learning-parameter, as approptiate, and `python main.py` to train Machie-Learning-Model. The result is saved a directory under [results](/results).
Then `python prediction.py` to interpolate Test-Dataset with the learned Machine-Learning-Model.<br>

Prepared [sample notebooks](/validate-notebook), which evaluates the learned Machine-Learning-Model compared to Linear, Spline, and Hermite interpolation.
The sample notebook [/validate-notebook/Plot_compare_methods.ipynb](/validate-notebook/Plot_compare_methods.ipynb) describes a comparison between ReLU and TanhExp for activation function of the Machine-Learning-Model.
The other sample notebook [/validate-notebook/Plot_scale_vs_data.ipynb](/validate-notebook/Plot_scale_vs_data.ipynb) shows a visualization of the actual interpolated Test-Dataset.