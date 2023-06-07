from mylib.learning_parameter import LearnParam
from mylib.get_save_directory import get_save_dir
from mylib.objected_model import Model
from scipy import interpolate as ip
from scipy import integrate as ig
import numpy as np
import copy
import pickle


def scalefactor_to_gyr(scalefactor):
    h = 0.6774
    y = lambda a: np.sqrt(a) / np.sqrt(0.3089 + 0.6911*np.power(a, 3))
    gyr = np.empty(scalefactor.shape)
    for i in range(gyr.shape[0]):
        for j in range(gyr.shape[1]):
            gyr[i, j] = ig.quad(y, 0.0, scalefactor[i, j])[0]

    return gyr * 10 / h


def divide_dataset(x = None, y = None, z = None, threshold = None):
    rowsize = 0
    colsize = 0
    if x is not None:
        rowsize = x.shape[0]
        colsize += 1
        abs_x = np.abs(x)
        abs_x = np.min(abs_x, axis = 1)
    if y is not None:
        rowsize = y.shape[0]
        colsize += 1
        abs_y = np.abs(y)
        abs_y = np.min(abs_y, axis = 1)
    if z is not None:
        rowsize = z.shape[0]
        colsize += 1
        abs_z = np.abs(z)
        abs_z = np.min(abs_z, axis = 1)

    abs_array = np.empty((rowsize, colsize))
    add_col = 0
    if x is not None:
        abs_array[:, add_col] = abs_x
        add_col += 1
    if y is not None:
        abs_array[:, add_col] = abs_y
        add_col += 1
    if z is not None:
        abs_array[:, add_col] = abs_z

    data1_indices, data2_indices = [], []
    for idx in range(abs_array.shape[0]):
        if np.any(abs_array[idx] <= threshold):
            data1_indices.append(idx)
        else:
            data2_indices.append(idx)

    print("length of the data1_indices : {0}".format(len(data1_indices)))
    print("length of the data2_indices : {0}".format(len(data2_indices)))

    return data1_indices, data2_indices


def interp(spline_y, spline_x, input_x, kind):
    interpolate = []
    for y, x, ip_x in zip(spline_y, spline_x, input_x):
        #print("x : {0}".format(x))
        #print("y : {0}".format(y))
        #print("ip_x : {0}".format(ip_x))
        func = ip.interp1d(x, y, kind = kind)
        interp = func(ip_x)
        interpolate.append(interp)

    return interpolate


def make_spline_dataset(data_input, scalefactor_input, scalefactor_output, input_size, rowsize):
    spline_y = []
    spline_x = []
    input_x = []
    for idx in range(len(rowsize)-1):
        if rowsize[idx+1] - rowsize[idx] <= 2:
            continue
        array_y = data_input[rowsize[idx]:rowsize[idx+1]]
        array_y1 = array_y[0]
        array_y2 = array_y[1:, input_size:].reshape(-1)
        spline_y.append(np.concatenate([array_y1, array_y2]))
        array_x = scalefactor_input[rowsize[idx]:rowsize[idx+1]]
        array_x1 = array_x[0]
        array_x2 = array_x[1:, input_size:].reshape(-1)
        spline_x.append(np.concatenate([array_x1, array_x2]))
        conc_x = scalefactor_output[rowsize[idx]:rowsize[idx+1]]
        array_input_x = np.concatenate([array_x[:, :input_size], conc_x], axis = 1).reshape(-1)
        array_input_x = np.concatenate([array_input_x, array_x[-1, input_size:]])
        input_x.append(array_input_x)

    return spline_y, spline_x, input_x


def make_hermite_dataset(rvir_input, rvir_output, xyz_input, v_input, scalefactor_input, scalefactor_output, input_size, rowsize):
    hermite_y = []
    hermite_dydx = []
    hermite_x = []
    input_x = []
    rvir_standardize = []
    for idx in range(len(rowsize)-1):
        if rowsize[idx+1] - rowsize[idx] <= 2:
            continue
        array_y = xyz_input[rowsize[idx]:rowsize[idx+1]] * rvir_input[rowsize[idx]:rowsize[idx+1]] / 1000
        array_y1 = array_y[0]
        array_y2 = array_y[1:, input_size:].reshape(-1)
        hermite_y.append(np.concatenate([array_y1, array_y2]))
        array_dydx = v_input[rowsize[idx]:rowsize[idx+1]]
        array_dydx1 = array_dydx[0]
        array_dydx2 = array_dydx[1:, input_size:].reshape(-1)
        hermite_dydx.append(np.concatenate([array_dydx1, array_dydx2]))
        array_x = scalefactor_input[rowsize[idx]:rowsize[idx+1]]
        array_x1 = array_x[0]
        array_x2 = array_x[1:, input_size:].reshape(-1)
        hermite_x.append(np.concatenate([array_x1, array_x2]))
        conc_x = scalefactor_output[rowsize[idx]:rowsize[idx+1]]
        array_input_x = np.concatenate([array_x[:, :input_size], conc_x], axis = 1).reshape(-1)
        array_input_x = np.concatenate([array_input_x, array_x[-1, input_size:]])
        input_x.append(array_input_x)
        array_rvir = rvir_input[rowsize[idx]:rowsize[idx+1]]
        conc_rvir = rvir_output[rowsize[idx]:rowsize[idx+1]]
        array_standardize_rvir = np.concatenate([array_rvir[:, :input_size], conc_rvir], axis = 1).reshape(-1)
        array_standardize_rvir = np.concatenate([array_standardize_rvir, array_rvir[-1, input_size:]])
        rvir_standardize.append(array_standardize_rvir)

    return hermite_y, hermite_dydx, hermite_x, input_x, rvir_standardize


def hermite(hermite_y, hermite_dydx, hermite_x, input_x, rvir_standardize):
    interpolate = []
    for y, dydx, x, ip_x, rvir in zip(hermite_y, hermite_dydx, hermite_x, input_x, rvir_standardize):
        func = ip.CubicHermiteSpline(x, y, dydx)
        interp = func(ip_x) / rvir * 1000
        interpolate.append(interp)
        
    return interpolate

def restore_dataset(data_input, data_output, input_size, rowsize):
    restore = []
    for idx in range(len(rowsize)-1):
        if rowsize[idx+1] - rowsize[idx] <= 2:
            continue
        array_input = data_input[rowsize[idx]:rowsize[idx+1]]
        array_output = data_output[rowsize[idx]:rowsize[idx+1]]
        conc_left = array_input[:, :input_size]
        array_restore = np.concatenate([conc_left, array_output], axis = 1).reshape(-1)
        array_restore = np.concatenate([array_restore, array_input[-1, input_size:]])
        restore.append(array_restore)

    return restore


def reshape_interpolation(interpolate, input_size, output_size):
    return_array_input = None
    return_array_output = None
    for data in interpolate:
        tmp_array = data[:-input_size].reshape(-1, input_size + output_size)
        output_array = tmp_array[:, input_size:]

        input_array = np.concatenate([tmp_array[:, :input_size], data[-input_size:].reshape(1, -1)], axis = 0)
        input_array = np.concatenate([input_array[:-1], input_array[1:]], axis = 1)

        if return_array_output is None:
            return_array_output = output_array
            return_array_input = input_array
        else:
            return_array_output = np.concatenate([return_array_output, output_array], axis = 0)
            return_array_input = np.concatenate([return_array_input, input_array], axis = 0)

    return return_array_input, return_array_output


if __name__ == "__main__":
    LP = LearnParam()
    ##load learned model
    directory_list = np.loadtxt("directory_list.txt", dtype = "str")
    for direc in range(len(directory_list)):
        print("{0}. {1}".format(direc, directory_list[direc]))

    direc = int(input("directory >> "))
    dirname = directory_list[direc]
    with open(dirname + "model.pickle", mode = "rb") as f:
        Model = pickle.load(f)

    ##load datasets
    with open(dirname + "test_input.pickle", mode = "rb") as f:
        test_input = pickle.load(f)
    with open(dirname + "test_output.pickle", mode = "rb") as f:
        test_output = pickle.load(f)
    with open(dirname + "test_input_scalefactor.pickle", mode = "rb") as f:
        test_input_scalefactor = pickle.load(f)
    with open(dirname + "test_output_scalefactor.pickle", mode = "rb") as f:
        test_output_scalefactor = pickle.load(f)
    with open(dirname + "test_input_rvir.pickle", mode = "rb") as f:
        test_input_rvir = pickle.load(f)
    with open(dirname + "test_output_rvir.pickle", mode = "rb") as f:
        test_output_rvir = pickle.load(f)
    with open(dirname + "test_rowsize.pickle", mode = "rb") as f:
        test_rowsize = pickle.load(f)

    for dim in range(test_input.shape[0]):
        print("dim = {0}, mean = {1}, std = {2}".format(dim, test_input[dim].mean(), test_input[dim].std()))

    ##interpolate
    print("Interpolating...")
    methods = ["origin", "model", "linear", "cubic"]
    if "x" in LP.use_param_input and "vx" in LP.use_param_input:
        if "hermite" not in methods:
            methods += ["hermite"]
    if "y" in LP.use_param_input and "vy" in LP.use_param_input:
        if "hermite" not in methods:
            methods += ["hermite"]
    if "z" in LP.use_param_input and "vz" in LP.use_param_input:
        if "hermite" not in methods:
            methods += ["hermite"]
    #methods = ["hermite"]
    interpolate = {}
    param_to_dim_input = {}
    param_to_dim_output = {}
    for dim, p_key in enumerate(LP.use_param_input):
        param_to_dim_input[p_key] = dim
    for dim, p_key in enumerate(LP.use_param_output):
        param_to_dim_output[p_key] = dim

    if "hermite" in methods:
        gyr_input = scalefactor_to_gyr(test_input_scalefactor)
        gyr_output = scalefactor_to_gyr(test_output_scalefactor)

    for p_key in LP.use_param_output:
        interpolate[p_key] = {}
        print("-----predict {0}-----".format(p_key))
        for met_key in methods:
            print("---{0}---".format(met_key))
            if met_key == "origin":
                data_input = test_input[param_to_dim_input[p_key]]
                data_output = test_output[param_to_dim_output[p_key]]
                print("data_input.shape : {0}".format(data_input.shape))
                print("data_output.shape : {0}".format(data_output.shape))
                interpolate[p_key][met_key] = restore_dataset(data_input, data_output, LP.INPUT_SIZE, test_rowsize)
                interpolate[p_key]["ScaleFactor"] = restore_dataset(test_input_scalefactor, test_output_scalefactor, LP.INPUT_SIZE, test_rowsize)
                print("length interpolate : {0}".format(len(interpolate[p_key][met_key])))
            if met_key == "model":
                model_predict = Model.predict(copy.deepcopy(test_input))
                data_input = test_input[param_to_dim_input[p_key]]
                data_output = model_predict[param_to_dim_output[p_key]]
                print("data_input.shape : {0}".format(data_input.shape))
                print("data_output.shape : {0}".format(data_output.shape))
                interpolate[p_key][met_key] = restore_dataset(data_input, data_output, LP.INPUT_SIZE, test_rowsize)
                print("length interpolate : {0}".format(len(interpolate[p_key][met_key])))
            if met_key == "linear" or met_key == "cubic":
                data_input = test_input[param_to_dim_input[p_key]]
                print("data_input.shape : {0}".format(data_input.shape))
                spline_y, spline_x, input_x = make_spline_dataset(data_input, test_input_scalefactor, test_output_scalefactor, LP.INPUT_SIZE, test_rowsize)
                interpolate[p_key][met_key] = interp(spline_y, spline_x, input_x, met_key)
                print("length interpolate : {0}".format(len(interpolate[p_key][met_key])))
            if met_key == "hermite":
                if not p_key in ["x", "y", "z"]:
                    continue
                elif not "v"+p_key in ["vx", "vy", "vz"]:
                    continue
                else:
                    xyz_input = test_input[param_to_dim_input[p_key]]
                    #v_input = test_input[param_to_dim_input["v"+p_key]] * 3.156 / 3.086 * 1e-3
                    v_input = test_input[param_to_dim_input["v"+p_key]]
                    hermite_y, hermite_dydx, hermite_x, input_x, rvir_standardize = make_hermite_dataset(test_input_rvir, test_output_rvir, xyz_input, v_input, gyr_input, gyr_output, LP.INPUT_SIZE, test_rowsize)
                    interpolate[p_key][met_key] = hermite(hermite_y, hermite_dydx, hermite_x, input_x, rvir_standardize)
                    print("length interpolate : {0}".format(len(interpolate[p_key][met_key])))

    ##save the interpolate
    with open(dirname + "interpolate.pickle", mode = "wb") as f:
        pickle.dump(interpolate, f)

    ##make interpolated arrays
    print("Make interpolated arrays...")
    if "x" in LP.use_param_output:
        input_x, _ = reshape_interpolation(interpolate["x"]["origin"], LP.INPUT_SIZE, LP.OUTPUT_SIZE)
    else:
        input_x = None
    if "y" in LP.use_param_output:
        input_y, _ = reshape_interpolation(interpolate["y"]["origin"], LP.INPUT_SIZE, LP.OUTPUT_SIZE)
    else:
        input_y = None
    if "z" in LP.use_param_output:
        input_z, _ = reshape_interpolation(interpolate["z"]["origin"], LP.INPUT_SIZE, LP.OUTPUT_SIZE)
    else:
        input_z = None
    data1_indices, data2_indices = divide_dataset(x = input_x, y = input_y, z = input_z, threshold = LP.THRESHOLD)
    
    interpolate_array = {}
    for p_key in LP.use_param_output:
        print("-----{0}-----".format(p_key))
        interpolate_array[p_key] = {0:{}, 1:{}}
        for met_key in methods:
            print("--{0}---".format(met_key))
            input_array, output_array = reshape_interpolation(interpolate[p_key][met_key], LP.INPUT_SIZE, LP.OUTPUT_SIZE)
            print(input_array.shape)
            print(output_array.shape)
            print(len(data1_indices))
            print(len(data2_indices))
            if not len(data1_indices) == 0:
                interpolate_array[p_key][0][met_key] = output_array[data1_indices]
            if not len(data2_indices) == 0:
                interpolate_array[p_key][1][met_key] = output_array[data2_indices]

    with open(dirname + "interpolate_array.pickle", mode = "wb") as f:
        pickle.dump(interpolate_array, f)
