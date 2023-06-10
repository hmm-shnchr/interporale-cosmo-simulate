class LearnParam:
    """
    Describe learning parameters of a MachineLearningModel.
    If you want to change some learning parameter,
    rewrite those variables.
    """

    def __init__(self):
        ## Simulation's box-size(Mpc/h)
        self.box_size               = 70
        self.train_mvir_threshold   = "1e+8"
        self.test_mvir_threshold    = "1e+7"
        self.extract_use_params     = ["ScaleFactor", "Mvir", "Rvir", "x", "vx", "host_Rvir"]
        self.use_param_input        = ["x", "vx", "Mvir"]
        self.use_param_output       = ["x"]
        self.input_size             = 2
        self.output_size            = 3
        ## "shift" or "equal_cpaced" can be specified for
        ## self.learn_dataset_format and self.predict_dataset_format
        self.learn_dataset_format   = "shift"
        self.predict_dataset_format = "equal_spaced"
        ## "All" or "Before_acc" or "After_acc" or "All_acc" or "All_not_acc" can be specified for self.extract_dataset
        self.extract_dataset        = "After_acc"