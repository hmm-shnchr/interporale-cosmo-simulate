def about_dir(save_dir, LP):
    """
    This function describes the conditions of learning model.
    """
    with open(save_dir + "AboutThisDirectory.txt", mode = "w") as f:
        ## threshold num
        line    = "threshold num : {}\n".format(LP.THRESHOLD)
        f.write(line)
        line    = "threshold of a training mvir : {}\n".format(LP.TRAIN_MVIR_THRESHOLD)
        f.write(line)
        line    = "threshold of a testing mvir : {}\n".format(LP.TEST_MVIR_THRESHOLD)
        f.write(line)
        ## parameters
        line    = "Input parameter : {}\n".format(LP.use_param_input)
        f.write(line)
        line    = "Output parameter : {}\n".format(LP.use_param_output)
        f.write(line)
        line    = "Add bias to dataset : {}\n".format(LP.add_bias)
        f.write(line)
        if LP.add_bias:
            line = "    The bias value : {}\n".format(LP.EPS)
            f.write(line)
        ## model
        line    = "Input size : {}, Output size : {}\n".format(LP.INPUT_SIZE, LP.OUTPUT_SIZE)
        f.write(line)
        line    = "Weight decay : "
        if LP.weight_decay:
            line += "True\n"
            line += "    Decay lambda : {}\n".format(LP.DECAY_LAMBDA)
        else:
            line += "False\n"
        f.write(line)
        line    = "Hidden layers : "
        neuron_variety = []
        for h_elem in LP.hidden:
            if len(neuron_variety) == 0 or neuron_variety[-1] != h_elem:
                neuron_variety.append(h_elem)
        cnt     = [1] * len(neuron_variety)
        cnt_i   = 0
        for i in range(1, len(LP.hidden)):
            if LP.hidden[i-1] == LP.hidden[i]:
                cnt[cnt_i] += 1
            else:
                cnt_i += 1
        for i in range(len(neuron_variety)):
            line += str(neuron_variety[i])
            if cnt[i] != 1:
                line += "*{}".format(cnt[i])
            if i != len(neuron_variety) - 1:
                line += " + "
        line    += "\n"
        f.write(line)
        line    = "Batch normalization : "
        if LP.batch_normalization:
            line += "True\n"
            f.write(line)
            if LP.batch_normalization_output:
                line = "    in output layer : True\n"
                f.write(line)
            else:
                line = "    in output layer : False\n"
                f.write(line)
        else:
            line += "False\n"
            f.write(line)
        line    = "Batch size : {}\n".format(LP.BATCH_SIZE)
        f.write(line)
        line    = "Activation function : {}\n".format(LP.activation_func)
        f.write(line)
        line    = "Weight initialize condition : {}\n".format(LP.weight_init)
        f.write(line)
        line    = "Lastlayer's activation function is identity : "
        if LP.lastlayer_identity:
            line += "True\n"
        else:
            line += "False\n"
        f.write(line)
        line    = "Loss function : {}\n".format(LP.loss_func)
        f.write(line)
        line    = "Optimizer : {}\n".format(LP.optimizer)
        f.write(line)
        line    = "Learning rate : {}\n".format(LP.LEARNING_RATE)
        f.write(line)
        line    = "Epoch : {}\n".format(LP.EPOCH)
        f.write(line)
        line    = "Normalization format of dataset : {}\n".format(LP.normalize_format)
        f.write(line)
        line    = "Extracted dataset : {}\n".format(LP.extract_dataset)
        f.write(line)
        line    = "Format of the learning dataset : {}\n".format(LP.learn_dataset_format)
        f.write(line)
        line    = "Format of the predict dataset : {}\n".format(LP.predict_dataset_format)
        f.write(line)