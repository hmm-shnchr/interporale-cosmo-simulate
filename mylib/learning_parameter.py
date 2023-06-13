import importlib

##True  : Use CuPy
##False : Not use CuPy(use NumPy)
use_cupy = False
def is_cupy():
    """
    Returns: bool: defined this file value.
    """
    return use_cupy
def xp_factory():
    """
    Returns: imported instance of CuPy or NumPy.
    """
    if is_cupy():
        return importlib.import_module("cupy")
    else:
        return importlib.import_module("numpy")

class LearnParam:
    """
    Describe learning parameter of a MachineLearningModel.
    If you want to change some learning parameter,
    rewrite those instance variables.

    Parameter
    ----------
    THRESHOLD: float
        Split dataset by this threshold.
        Model[0] learns dataset below this threshold.
        Model[1] learns dataset above this threshold.

    EPS: float
        Prevent division by 0.

    accuracy_test: bool
        Whether to print Accuracy-Value for Test-Dataset every epoch.

    accuracy_batch: bool
        Whether to print Accuracy-Value for Minibatch every epoch.

    TRAIN_MVIR_THRESHOLD: string
        Use halos with Mvir(z=0) greater than this value for Train-Dataset

    TEST_MVIR_THRESHOLD: string
        Use halos with Mvir(z=0) greater than this value for Test-Dataset

    use_param_input: list
        

    """

    def __init__(self):
        self.THRESHOLD                  = 0.03
        self.EPS                        = 1e-7
        self.accuracy_test              = True
        self.accuracy_batch             = True
        self.TRAIN_MVIR_THRESHOLD       = "1e+8"
        self.TEST_MVIR_THRESHOLD        = "1e+7"
        self.use_param_input            = ["x", "vx", "Mvir"]
        self.use_param_output           = ["x"]
        self.add_bias                   = False
        self.weight_decay               = False
        self.DECAY_LAMBDA               = 1e-3
        self.hidden                     = [100]*10
        self.BATCH_SIZE                 = 50
        self.LEARNING_RATE              = "1e-3"
        self.optimizer                  = "Adam"
        self.batch_normalization        = True
        self.batch_normalization_output = False
        self.loss_func                  = "RE"
        self.activation_func            = "tanhexp"
        self.weight_init                = "he"
        self.lastlayer_identity         = True
        self.EPOCH                      = 1000
        self.INPUT_SIZE                 = 2
        self.OUTPUT_SIZE                = 3
        self.normalize_format           = "Standardization"
        self.learn_dataset_format       = "shift"
        self.predict_dataset_format     = "equal_spaced"
        self.extract_dataset            = "After_acc"
        self.SPLIT_EPOCH                = 10
        self.LEARN_NUM                  = 1
        self.SAVE_FIG_TYPE              = ".png"