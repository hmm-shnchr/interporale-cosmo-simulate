from mylib.learning_parameter   import LearnParam
from mylib.get_save_directory   import get_save_dir
from mylib.objected_model       import Model
from mylib.about_directory      import about_dir
import os, sys
import copy
import pickle


if __name__ == "__main__":
    LP = LearnParam()
    save_dir = get_save_dir(LP)
    if os.path.isdir(save_dir):
        print("\nWarning!!!\nDirectory \"{save_dir}\" is an existing directory.\nIt will be overwritten.".format(save_dir=save_dir))
        print("1. Continue.\n2. Exit.")
        while True:
            cont = int(input(">> "))
            if cont == 1:
                print("Continue.")
                break
            elif cont == 2:
                print("Exit.")
                sys.exit(1)
            else: pass

    print("Input parameters : {}".format(LP.use_param_input))
    print("Output parameters : {}".format(LP.use_param_output))

    fname = "./make-dataset/{0}in{1}out".format(LP.INPUT_SIZE, LP.OUTPUT_SIZE)
    for p_key in LP.use_param_input:
        fname += "_" + p_key
    fname += "_to"
    for p_key in LP.use_param_output:
        fname += "_" + p_key
    fname += "/"
    if not os.path.isdir(fname):
        print("{dir} does not exist.".format(dir=fname))
        sys.exit(1)

    with open(fname + LP.TRAIN_MVIR_THRESHOLD + "_train_input.pickle", mode = "rb") as f:
        train_input = pickle.load(f)
    with open(fname + LP.TRAIN_MVIR_THRESHOLD + "_train_output.pickle", mode = "rb") as f:
        train_output = pickle.load(f)
    with open(fname + LP.TEST_MVIR_THRESHOLD + "_test_input.pickle", mode = "rb") as f:
        test_input = pickle.load(f)
    with open(fname + LP.TEST_MVIR_THRESHOLD + "_test_output.pickle", mode = "rb") as f:
        test_output = pickle.load(f)
    with open(fname + LP.TEST_MVIR_THRESHOLD + "_test_input_scalefactor.pickle", mode = "rb") as f:
        test_input_scalefactor = pickle.load(f)
    with open(fname + LP.TEST_MVIR_THRESHOLD + "_test_output_scalefactor.pickle", mode = "rb") as f:
        test_output_scalefactor = pickle.load(f)
    with open(fname + LP.TEST_MVIR_THRESHOLD + "_test_input_rvir.pickle", mode = "rb") as f:
        test_input_rvir = pickle.load(f)
    with open(fname + LP.TEST_MVIR_THRESHOLD + "_test_output_rvir.pickle", mode = "rb") as f:
        test_output_rvir = pickle.load(f)
    with open(fname + LP.TEST_MVIR_THRESHOLD + "_test_rowsize.pickle", mode = "rb") as f:
        test_rowsize = pickle.load(f)
    print("train_input.shape : {0}".format(train_input.shape))
    print("train_output.shape : {0}".format(train_output.shape))
    print("test_input.shape : {0}".format(test_input.shape))
    print("test_output.shape : {0}".format(test_output.shape))

    param_to_dim_input  = {}
    for dim, p_key in enumerate(LP.use_param_input):
        param_to_dim_input[p_key]   = dim
    param_to_dim_output = {}
    for dim, p_key in enumerate(LP.use_param_output):
        param_to_dim_output[p_key]  = dim
    input_dim   = len(LP.use_param_input)
    output_dim  = len(LP.use_param_output)
    print("param_to_dim_input : {0}\nparam_to_dim_output : {1}".format(param_to_dim_input, param_to_dim_output))

    Model = Model(LP.INPUT_SIZE, input_dim,
                  LP.hidden, LP.activation_func, LP.weight_init,
                  LP.batch_normalization, LP.batch_normalization_output,
                  LP.OUTPUT_SIZE, output_dim, LP.lastlayer_identity,
                  LP.loss_func, LP.weight_decay, LP.DECAY_LAMBDA,
                  LP.SPLIT_EPOCH, BATCH_SIZE_MAX = 200000)

    dp = lambda x: copy.deepcopy(x)
    Model.learning(dp(train_input), dp(train_output),
                   dp(test_input), dp(test_output),
                   param_to_dim_input, param_to_dim_output,
                   LP.optimizer, LP.LEARNING_RATE,
                   LP.BATCH_SIZE, LP.EPOCH,
                   LP.normalize_format, LP.THRESHOLD,
                   LP.accuracy_test, LP.accuracy_batch)

    ## Make a directory to save results.
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print("Make a directory {}.".format(save_dir))

    ## Plot the Loss function and Accuracy values for each epoch.
    Model.plot_figures(save_dir, LP.SAVE_FIG_TYPE)
    print("Plot Loss and Accuracy.")

    with open(save_dir + "model.pickle", mode = "wb") as f:
        pickle.dump(Model, f)
        print("Saved the learned model.")

    about_dir(save_dir, LP)

    ##Add the directory where you saved the results to directory_list.txt
    with open("directory_list.txt", mode = "a") as f:
        f.writelines(save_dir + "\n")
        print("Append \"{}\" to directory_list.txt.".format(save_dir))

    with open(save_dir + "train_input.pickle", mode = "wb") as f:
        pickle.dump(train_input, f)
    with open(save_dir + "train_output.pickle", mode = "wb") as f:
        pickle.dump(train_output, f)
    with open(save_dir + "test_input.pickle", mode = "wb") as f:
        pickle.dump(test_input, f)
    with open(save_dir + "test_output.pickle", mode = "wb") as f:
        pickle.dump(test_output, f)
    with open(save_dir + "test_input_scalefactor.pickle", mode = "wb") as f:
        pickle.dump(test_input_scalefactor, f)
    with open(save_dir + "test_output_scalefactor.pickle", mode = "wb") as f:
        pickle.dump(test_output_scalefactor, f)
    with open(save_dir + "test_input_rvir.pickle", mode = "wb") as f:
        pickle.dump(test_input_rvir, f)
    with open(save_dir + "test_output_rvir.pickle", mode = "wb") as f:
        pickle.dump(test_output_rvir, f)
    with open(save_dir + "test_rowsize.pickle", mode = "wb") as f:
        pickle.dump(test_rowsize, f)