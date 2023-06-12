# make-dataset

`python make_dataset_align_scalefactor.py` makes Train-Dataset and Test-Dataset. To supervised learning, these datasets have Input-Data and Correct-Label. Parameters used for learning Machine-Learning-Model is defined by [learning_parameter.py](/make-dataset/learning_parameter.py).
<br>

Datasets is created by separating subhalo with Mvir(z=0) greater than ***train_mvir_threshold***(Train-Dataset), ***test_mvir_threshold***(Test-Dataset) and others, and Machine-Learning-Models are created for each. Created 2 Machine-Learning-Model learn these separated datasets.

Parameters used for learning and normalizing them are defined by ***extract_use_params***.

Parameters used for Input-Data and Correct-Label are specified ***use_param_input*** and ***use_param_output***.
These parameters are must included in ***extract_use_paras***.

Datasets are 3-dimensional matrix, each dimension is *[parameter, dataset size, input(output) size]*.
These *input size* and *output size* are defined by ***input_size*** and ***output_size***.

Format of datasets are defined by ***learn_dataset_format***(Train-Dataset) and ***predict_dataset_format***(Test-Dataset).
*"shift"* or *""equal_spaced"* can be specified for these valiables. Examples of *"shift"* and *""equal_spaced"* are shown below.

<!-- 1. *"shift"*   -->
When data is like $[x_1, x_2, \dots, x_n]$ and (***input_size***, ***output_size***) is (2, 3),  

$$
{\left\lbrack \matrix{2 & 3 \cr 4 & 5} \right\rbrack}* \left\lbrack \matrix{1 & 0 \cr 0 & 1} \right\rbrack
= \left\lbrack \matrix{2 & 3 \cr 4 & 5} \right\rbrack
$$


<!-- 2. *""equal_spaced"*   -->
When data is like $[x_1, x_2, \dots, x_n]$ and (***input_size***, ***output_size***) is (2, 3),  