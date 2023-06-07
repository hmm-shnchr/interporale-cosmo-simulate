def get_save_dir(LP):
    dirname = "results/"
    dirname += str(LP.INPUT_SIZE) + "in_" + str(LP.OUTPUT_SIZE) + "out/"
    for param in LP.use_param_input:
        if param != "time_idx":
            dirname += param + "_"
    dirname += "to_"
    for param in LP.use_param_output:
        dirname += param + "_"
    dirname += str(len(LP.hidden)) + "layers_" + str(LP.hidden[0]) + "neurons_"
    if LP.batch_normalization:
        dirname += "BatchNorm_"
    dirname += LP.loss_func + "_"
    dirname += LP.LEARNING_RATE + "lr_"
    dirname += str(LP.EPOCH) + "Epoch_"
    if LP.normalize_format != "None":
        dirname += LP.normalize_format + "_"
    dirname += LP.extract_dataset + "_"
    dirname += "trainMvir" + LP.TRAIN_MVIR_THRESHOLD + "_"
    dirname += "testMvir" + LP.TEST_MVIR_THRESHOLD + "_"
    dirname += str(LP.LEARN_NUM) + "/"

    return dirname