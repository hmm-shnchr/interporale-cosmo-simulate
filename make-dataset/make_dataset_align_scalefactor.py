from tqdm import tqdm
from learning_parameter import LearnParam
from extract_dataset import extract_dataset
import numpy as np
import pickle
import os


class ReshapeDataset:

    def __init__(self, dtype = np.float32):
        self.dtype          = dtype
        self.input_size     = None
        self.output_size    = None
        self.rowsize        = [0]

    def reshape_dataset(self, dataset, input_size, output_size, mainbranch, mode):
        self.input_size     = input_size
        self.output_size    = output_size

        input_return    = {}
        output_return   = {}
        unit_size       = input_size + output_size + input_size
        for m_key in mainbranch:
            input_return[m_key]     = None
            output_return[m_key]    = None
            for data in tqdm(dataset[m_key]):
                if data.shape[2] < unit_size: continue

                if mode == "shift":
                    rowsize     = data.shape[2] - unit_size + 1
                    tmp_array   = np.empty((data.shape[0], rowsize, unit_size))
                    for col in range(unit_size):
                        tmp_array[:, :, col] = data[:, 0, col:col + rowsize]
                    array_input     = np.concatenate([tmp_array[:, :, :self.input_size], tmp_array[:, :, self.input_size + self.output_size:]], axis = 2)
                    array_output    = tmp_array[:, :, self.input_size:self.input_size + self.output_size]
                    if input_return[m_key] is None:
                        input_return[m_key] = array_input
                    else:
                        input_return[m_key] = np.concatenate([input_return[m_key], array_input], axis = 1)
                    if output_return[m_key] is None:
                        output_return[m_key] = array_output
                    else:
                        output_return[m_key] = np.concatenate([output_return[m_key], array_output], axis = 1)

                if mode == "equal_spaced":
                    use_size    = int(data.shape[2] / (unit_size - input_size))
                    use_size    = use_size * (unit_size - input_size) + input_size
                    if data.shape[2] - use_size < 0: use_size -= (unit_size - input_size)
                    use_data    = data[:, 0, -use_size:]
                    tmp_array   = use_data[:, :-input_size].reshape(use_data.shape[0], -1, input_size + output_size)

                    if self.rowsize[-1] == 0:
                        self.rowsize.append(tmp_array.shape[1])
                    else:
                        self.rowsize.append(self.rowsize[-1] + tmp_array.shape[1])

                    array_output    = tmp_array[:, :, input_size:]
                    array_input1    = tmp_array[:, :, :input_size]
                    array_input2    = np.concatenate([array_input1[:, 1:], use_data[:, -input_size:].reshape(use_data.shape[0], 1, -1)], axis = 1)
                    array_input     = np.concatenate([array_input1, array_input2], axis = 2)

                    if input_return[m_key] is None:
                        input_return[m_key] = array_input
                    else:
                        input_return[m_key] = np.concatenate([input_return[m_key], array_input], axis = 1)

                    if output_return[m_key] is None:
                        output_return[m_key] = array_output
                    else:
                        output_return[m_key] = np.concatenate([output_return[m_key], array_output], axis = 1)

        _input_return   = None
        _output_return  = None
        for m_key in mainbranch:
            if _input_return is None:
                _input_return = input_return[m_key]
            else:
                _input_return = np.concatenate([_input_return, input_return[m_key]], axis = 1)

            if _output_return is None:
                _output_return = output_return[m_key]
            else:
                _output_return = np.concatenate([_output_return, output_return[m_key]], axis = 1)

        return _input_return, _output_return


if __name__ == "__main__":
    LP = LearnParam()
    print("Loading datasets...")
    host_param          = {}
    sub_param           = {}
    dname_train_datas   = "../make-pickle-mergertree/sample-params/"
    dname_test_datas    = "../make-pickle-mergertree/sample-params/"
    with open(dname_train_datas + "host_param.pickle", mode = "rb") as f:
        host_param["train"] = pickle.load(f)
    with open(dname_train_datas + "sub_param.pickle", mode = "rb") as f:
        sub_param["train"]  = pickle.load(f)
    with open(dname_test_datas + "host_param.pickle", mode = "rb") as f:
        host_param["test"]  = pickle.load(f)
    with open(dname_test_datas + "sub_param.pickle", mode = "rb") as f:
        sub_param["test"]   = pickle.load(f)

    ## Get aligned data.
    ## Alige a training-dataset's time resolution to testing-dataset.
    print("Get aligned datasets...")
    set_type    = "train"
    mainbranch  = list(sub_param[set_type].keys())
    parameter   = list(sub_param[set_type][mainbranch[0]].keys())
    for m_key in mainbranch:
        for p_key in parameter:
            if not p_key in ["ax", "ay", "az"]:
                h_p = host_param[set_type][m_key][p_key]
                h_p = h_p[::-2]
                host_param[set_type][m_key][p_key] = h_p[::-1]
            for s_idx, s_p in enumerate(sub_param[set_type][m_key][p_key]):
                s_p = s_p[::-2]
                s_p = s_p[::-1]
                sub_param[set_type][m_key][p_key][s_idx] = s_p

    ## Get accretion time.
    print("Get accretion time...")
    mainbranch      = {}
    parameter       = {}
    scalefactor_acc = {}
    sets            = ["train", "test"]
    for set_type in sets:
        mainbranch[set_type]        = list(sub_param[set_type].keys())
        parameter[set_type]         = list(sub_param[set_type][mainbranch[set_type][0]].keys())
        print("===== parameter[{set_type}] =====".format(set_type=set_type))
        print(parameter[set_type])
        scalefactor_acc[set_type]   = {}
        for m_key in mainbranch[set_type]:
            scalefactor_acc[set_type][m_key] = []
            host_id = host_param[set_type][m_key]["ID"]
            for sub_idx, pid in enumerate(sub_param[set_type][m_key]["pid"]):
                acc_time = np.where(host_id[-pid.size:] == pid)
                if acc_time[0].size == 0:
                    scalefactor_acc[set_type][m_key].append(-1)
                else:
                    scalefactor_acc[set_type][m_key].append(acc_time[0][0])

    ## Add sub_param to host_param's Rvir.
    for set_type in sets:
        parameter[set_type].append("host_Rvir")
        for m_key in mainbranch[set_type]:
            sub_param[set_type][m_key]["host_Rvir"] = []
            for sub_p in sub_param[set_type][m_key][parameter[set_type][0]]:
                sub_param[set_type][m_key]["host_Rvir"].append(host_param[set_type][m_key]["Rvir"][-sub_p.size:])

    ## Extract using datas.
    print("Extract using datas...")
    _dataset = {}
    for set_type in sets:
        if set_type == "train":
            threshold = float(LP.train_mvir_threshold)
        if set_type == "test":
            threshold = float(LP.test_mvir_threshold)
        _dataset[set_type] = extract_dataset(mainbranch[set_type], parameter[set_type],
                                             host_param[set_type], sub_param[set_type],
                                             LP.extract_dataset, scalefactor_acc[set_type],
                                             LP.input_size, LP.output_size, threshold, LP.box_size)

    _exclude_parameter  = ["ID", "pid", "upid"]
    exclude_parameter   = []
    for exc_p in _exclude_parameter:
        if exc_p in parameter[sets[1]]: exclude_parameter.append(exc_p)

    dataset = {}
    for set_type in sets:
        dataset[set_type] = {}
        for m_key in mainbranch[set_type]:
            dataset[set_type][m_key] = []
            for idx in range(len(_dataset[set_type][m_key][parameter[set_type][0]])):
                data_np = []
                for p_key in parameter[set_type]:
                    if p_key in exclude_parameter: continue
                    data = _dataset[set_type][m_key][p_key][idx].astype(np.float32)
                    data_np.append(data)
                data_np = np.array(data_np)
                data_np = data_np.reshape(len(parameter[set_type]) - len(exclude_parameter), 1, -1)
                dataset[set_type][m_key].append(data_np)

    ## Extract use parameters.
    parameter       = LP.extract_use_params
    param_to_dim    = {}
    for dim, p_key in enumerate(parameter):
        param_to_dim[p_key] = dim
    use_parameter   = LP.use_param_input
    if "ScaleFactor" in use_parameter:  use_parameter.remove("ScaleFactor")
    if "host_Rvir" in use_parameter:  use_parameter.remove("host_Rvir")
    print("use_parameter : {0}".format(use_parameter))
    use_parameter_indices = []
    for p_key in use_parameter + ["ScaleFactor", "host_Rvir"]:
        use_parameter_indices.append(param_to_dim[p_key])
    print("use_parameter_indices : {0}".format(use_parameter_indices))

    ## Extract  dimensions of use parameters.
    print("Extract using dimensions...")
    for set_type in sets:
        for m_key in mainbranch[set_type]:
            for sub_idx, sub_params in enumerate(dataset[set_type][m_key]):
                dataset[set_type][m_key][sub_idx] = sub_params[use_parameter_indices, ...]

    ## Reshape dataset.
    RD = ReshapeDataset()
    print("Make a training dataset...")
    train_input, train_output   = RD.reshape_dataset(dataset["train"],
                                                     LP.input_size, LP.output_size,
                                                     mainbranch["train"],
                                                     mode=LP.learn_dataset_format)
    print("Make a testing dataset...")
    test_input, test_output     = RD.reshape_dataset(dataset["test"],
                                                     LP.input_size, LP.output_size,
                                                     mainbranch["test"],
                                                     mode=LP.predict_dataset_format)
    rowsize_test                = RD.rowsize

    input_dim       = []
    output_dim      = []
    dim_scalefactor = -2    ## TODO calc
    dim_rvir        = -1    ## TODO calc
    for dim, p_key in enumerate(use_parameter):
        if p_key in LP.use_param_input:     input_dim.append(dim)
        if p_key in LP.use_param_output:    output_dim.append(dim)
    print("input_dim : {0}".format(input_dim))
    print("output_dim : {0}".format(output_dim))

    train_input_scalefactor     = train_input[dim_scalefactor]
    test_input_scalefactor      = test_input[dim_scalefactor]
    train_output_scalefactor    = train_output[dim_scalefactor]
    test_output_scalefactor     = test_output[dim_scalefactor]
    test_input_rvir             = test_input[dim_rvir]
    test_output_rvir            = test_output[dim_rvir]
    train_input                 = train_input[input_dim]
    test_input                  = test_input[input_dim]
    train_output                = train_output[output_dim]
    test_output                 = test_output[output_dim]
    print("train_input.shape : {0}".format(train_input.shape))
    print("train_output.shape : {0}".format(train_output.shape))
    print("test_input.shape : {0}".format(test_input.shape))
    print("test_output.shape : {0}".format(test_output.shape))
    print("train_input_scalefactor.shape : {0}".format(train_input_scalefactor.shape))
    print("train_output_scalefactor.shape : {0}".format(train_output_scalefactor.shape))
    print("test_input_scalefactor.shape : {0}".format(test_input_scalefactor.shape))
    print("test_output_scalefactor.shape : {0}".format(test_output_scalefactor.shape))
    print("test_input_rvir.shape : {0}".format(test_input_rvir.shape))
    print("test_output_rvir.shape : {0}".format(test_output_rvir.shape))
    print("rowsize_test.length : {0}".format(len(rowsize_test)))

    dname = "{0}in{1}out".format(LP.input_size, LP.output_size)
    for p_key in LP.use_param_input:
        dname += "_" + p_key
    dname += "_to"
    for p_key in LP.use_param_output:
        dname += "_" + p_key
    if not os.path.isdir(dname):
        os.mkdir(dname)
    dname += "/"
    with open(dname + LP.train_mvir_threshold + "_train_input.pickle", mode = "wb") as f:
        pickle.dump(train_input, f)
    with open(dname + LP.train_mvir_threshold + "_train_output.pickle", mode = "wb") as f:
        pickle.dump(train_output, f)
    with open(dname + LP.test_mvir_threshold + "_test_input.pickle", mode = "wb") as f:
        pickle.dump(test_input, f)
    with open(dname + LP.test_mvir_threshold + "_test_output.pickle", mode = "wb") as f:
        pickle.dump(test_output, f)
    with open(dname + LP.train_mvir_threshold + "_train_input_scalefactor.pickle", mode = "wb") as f:
        pickle.dump(train_input_scalefactor, f)
    with open(dname + LP.train_mvir_threshold + "_train_output_scalefactor.pickle", mode = "wb") as f:
        pickle.dump(train_output_scalefactor, f)
    with open(dname + LP.test_mvir_threshold + "_test_input_scalefactor.pickle", mode = "wb") as f:
        pickle.dump(test_input_scalefactor, f)
    with open(dname + LP.test_mvir_threshold + "_test_output_scalefactor.pickle", mode = "wb") as f:
        pickle.dump(test_output_scalefactor, f)
    with open(dname + LP.test_mvir_threshold + "_test_input_rvir.pickle", mode = "wb") as f:
        pickle.dump(test_input_rvir, f)
    with open(dname + LP.test_mvir_threshold + "_test_output_rvir.pickle", mode = "wb") as f:
        pickle.dump(test_output_rvir, f)
    with open(dname + LP.test_mvir_threshold + "_test_rowsize.pickle", mode = "wb") as f:
        pickle.dump(rowsize_test, f)