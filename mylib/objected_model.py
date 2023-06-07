from tqdm import tqdm
from mylib.machine_learning_model import MachineLearningModel
from mylib.normalization import Normalization
from mylib.optimizer import set_optimizer
from mylib.layers import *
import matplotlib.pyplot as plt
import numpy as np
import mylib.learning_parameter as LP
if LP.use_cupy:
    import cupy


class Model:
    def __init__(self, input_size, input_dim,
                 hidden, act_func, weight_init, batch_norm, batch_norm_output,
                 output_size, output_dim, lastlayer_identity,
                 loss_func, weight_decay, decay_lambda, split_epoch,
                 acc_func = "RelativeError", dtype = np.float32, BATCH_SIZE_MAX = 20000):
        self.input_size, self.input_dim = input_size, input_dim
        self.hidden, self.act_func, self.weight_init = hidden, act_func, weight_init
        self.batch_norm, self.batch_norm_output = batch_norm, batch_norm_output
        self.output_size, self.output_dim = output_size, output_dim
        self.lastlayer_identity = lastlayer_identity
        self.loss_func = loss_func
        self.weight_decay, self.decay_lambda = weight_decay, decay_lambda
        self.acc_func = acc_func
        self.dtype = dtype
        self.batch_max = BATCH_SIZE_MAX
        self.batch_size = 100
        self.norm_format = None
        self.Norm_input = None
        self.Norm_output = None
        self.model = {}
        self.threshold = None
        self.optimizer = None
        self.loss_val = {}
        self.train_acc, self.test_acc = {}, {}
        self.param_to_dim_input = None
        self.param_to_dim_output = None
        self.is_acc_test = None
        self.split_epoch = split_epoch
        self.iter_nums = None


    def __div_dataset(self, x = None, y = None, z = None):
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
            if np.any(abs_array[idx] <= self.threshold):
                data1_indices.append(idx)
            else:
                data2_indices.append(idx)

        print("length of the data1_indices : {0}".format(len(data1_indices)))
        print("length of the data2_indices : {0}".format(len(data2_indices)))

        return data1_indices, data2_indices


    def learning(self, _train_input, _train_output,
                 _test_input, _test_output,
                 param_to_dim_input, param_to_dim_output,
                 opt, lr,
                 batch_size, epoch,
                 norm_format, threshold = 0.03,
                 is_acc_test = True, is_acc_batch = True):
        self.is_acc_test = is_acc_test
        self.is_acc_batch = is_acc_batch
        self.threshold = threshold
        self.batch_size = batch_size

        print("Divide training dataset...")
        train_indices = {}
        input_x = _train_input[param_to_dim_input["x"]] if "x" in param_to_dim_input.keys() else None
        input_y = _train_input[param_to_dim_input["y"]] if "y" in param_to_dim_input.keys() else None
        input_z = _train_input[param_to_dim_input["z"]] if "z" in param_to_dim_input.keys() else None
        train_indices[0], train_indices[1] = self.__div_dataset(x = input_x, y = input_y, z = input_z)

        print("Divide testing dataset...")
        test_indices = {}
        input_x = _test_input[param_to_dim_input["x"]] if "x" in param_to_dim_input.keys() else None
        input_y = _test_input[param_to_dim_input["y"]] if "y" in param_to_dim_input.keys() else None
        input_z = _test_input[param_to_dim_input["z"]] if "z" in param_to_dim_input.keys() else None
        test_indices[0], test_indices[1] = self.__div_dataset(x = input_x, y = input_y, z = input_z)

        train_input, train_output = {}, {}
        test_input, test_output = {}, {}
        iter_nums = []
        for num in range(2):
            if len(train_indices[num]) == 0 and len(test_indices[num]) == 0:
                continue
            train_input[num] = _train_input[:, train_indices[num]]
            train_output[num] = _train_output[:, train_indices[num]]
            test_input[num] = _test_input[:, test_indices[num]]
            test_output[num] = _test_output[:, test_indices[num]]
            iter_nums.append(num)

        self.norm_format = norm_format
        Norm_input = {}
        for key, dim in param_to_dim_input.items():
            if key in ["ScaleFactor"]:
                continue
            Norm_input[key] = {}
            print("\n-----Input parameter : {0}-----".format(key))
            for num in iter_nums:
                Norm_input[key][num] = Normalization(self.norm_format)
                train_input[num][dim, ...] = Norm_input[key][num].run(train_input[num][dim, ...])
                if is_acc_test:
                    test_input[num][dim, ...] = Norm_input[key][num].run_predict(test_input[num][dim, ...])

        Norm_output = {}
        for key, dim in param_to_dim_output.items():
            if key in ["ScaleFactor"]:
                continue
            Norm_output[key] = {}
            print("\n-----Output parameter : {0}-----".format(key))
            for num in iter_nums:
                Norm_output[key][num] = Normalization(self.norm_format)
                train_output[num][dim, ...] = Norm_output[key][num].run(train_output[num][dim, ...])
                if is_acc_test:
                    test_output[num][dim, ...] = Norm_output[key][num].run_predict(test_output[num][dim, ...])

        self.Norm_input = Norm_input
        self.Norm_output = Norm_output
        optimizer = {}

        for num in iter_nums:
            print("\n-----start learning of the model : {0}-----".format(num))
            print("train_input.shape : {0}".format(train_input[num].shape))
            print("train_output.shape : {0}".format(train_output[num].shape))
            print("test_input.shape : {0}".format(test_input[num].shape))
            print("test_output.shape : {0}".format(test_output[num].shape))
            rowsize_train = train_input[num].shape[1]
            batch_mask_arange = np.arange(rowsize_train)
            batch_size = int(rowsize_train / self.batch_size)
            iter_per_epoch = int(rowsize_train / batch_size)
            iter_num = iter_per_epoch * epoch
            print("\nBatch size : {}\nIterations per 1epoch : {}\nIterations : {}\nEpoch : {}".format(batch_size, iter_per_epoch, iter_num, epoch))

            self.model[num] = MachineLearningModel(self.input_size, self.input_dim,
                                                   self.hidden, self.act_func, self.weight_init, self.batch_norm, self.batch_norm_output,
                                                   self.output_size, self.output_dim, self.lastlayer_identity,
                                                   self.loss_func, self.weight_decay, self.decay_lambda)
            optimizer[num] = set_optimizer(opt, float(lr))

            train_batch = None
            if train_input[num].shape[1] > self.batch_max:
                train_batch = np.arange(0, train_input[num].shape[1], self.batch_max).tolist()
                if train_batch[-1] != train_input[num].shape[1] - 1:
                    train_batch.append(train_input[num].shape[1])

            if is_acc_test:
                if is_acc_batch == False:
                    test_batch = None
                    if test_input[num].shape[1] > self.batch_max:
                        test_batch = np.arange(0, test_input[num].shape[1], self.batch_max).tolist()
                        if test_batch[-1] != test_input[num].shape[1]:
                            test_batch.append(test_input[num].shape[1])
                else:
                    test_batch = int(self.batch_max / 10)

            for key in param_to_dim_output.keys():
                self.loss_val[key + str(num)] = []
                self.train_acc[key + str(num)] = []
                if is_acc_test:
                    self.test_acc[key + str(num)] = []

            for i in tqdm(range(iter_num)):
                batch_mask = np.random.choice(batch_mask_arange, batch_size)
                batch_input = train_input[num][:, batch_mask]
                batch_output = train_output[num][:, batch_mask]
                if LP.use_cupy:
                    batch_input = cupy.asarray(batch_input)
                    batch_output = cupy.asarray(batch_output)

                grads = self.model[num].gradient(batch_input, batch_output, is_training = True)
                model_params = self.model[num].params
                optimizer[num].update(model_params, grads)
                if i % (iter_per_epoch * self.split_epoch) == 0:
                    loss = self.model[num].loss(batch_input, batch_output, is_training = False)

                    if LP.use_cupy:
                        ##use cupy
                        loss = cupy.asnumpy(loss)
                        if train_batch is None:
                            train_acc = self.model[num].accuracy(cupy.asarray(train_input[num]),
                                                                 cupy.asarray(train_output[num]),
                                                                 acc_func = self.acc_func, is_training = False)
                        else:
                            train_acc = 0.0
                            for j in range(len(train_batch) - 1):
                                batch_input = cupy.asarray(train_input[num][:, train_batch[j]:train_batch[j+1]])
                                batch_output = cupy.asarray(train_output[num][:, train_batch[j]:train_batch[j+1]])
                                train_acc += self.model[num].accuracy(batch_input, batch_output,
                                                                      acc_func = self.acc_func, is_training = False)
                            train_acc /= len(train_batch) - 1
                        train_acc = cupy.asnumpy(train_acc)

                        if is_acc_test:
                            if test_batch is None:
                                test_acc = self.model[num].accuracy(cupy.asarray(test_input[num]),
                                                                    cupy.asarray(test_output[num]),
                                                                    acc_func = self.acc_func, is_training = False)
                            elif is_acc_batch == False:
                                test_acc = 0.0
                                for j in range(len(test_batch) - 1):
                                    batch_input = cupy.asarray(test_input[num][:, test_batch[j]:test_batch[j+1]])
                                    batch_output = cupy.asarray(test_output[num][:, test_batch[j]:test_batch[j+1]])
                                test_acc /= len(test_batch) - 1
                            else:
                                batch_mask = np.random.choice(np.arange(test_input[num].shape[1]), test_batch)
                                batch_input = cupy.asarray(test_input[num][:, batch_mask])
                                batch_output = cupy.asarray(test_output[num][:, batch_mask])
                                test_acc = self.model[num].accuracy(batch_input, batch_output,
                                                                    acc_func = self.acc_func, is_training = False)
                            test_acc = cupy.asnumpy(test_acc)

                    else:
                        ##not use cupy
                        train_acc = self.model[num].accuracy(train_input[num], train_output[num],
                                                             acc_func = self.acc_func, is_training = False)
                        if is_acc_test:
                            test_acc = self.model[num].accuracy(test_input[num], test_output[num],
                                                                acc_func = self.acc_func, is_training = False)

                    for p_key, dim in param_to_dim_output.items():
                        self.loss_val[p_key + str(num)].append(loss[dim])
                        self.train_acc[p_key + str(num)].append(train_acc[dim])
                        if is_acc_test:
                            self.test_acc[p_key + str(num)].append(test_acc[dim])
                        else:
                            pass

        self.optimizer = optimizer
        self.param_to_dim_input = param_to_dim_input
        self.param_to_dim_output = param_to_dim_output
        self.iter_nums = iter_nums


    def predict(self, _data_input):
        data_indices = {}
        input_x = _data_input[self.param_to_dim_input["x"]] if "x" in self.param_to_dim_input.keys() else None
        input_y = _data_input[self.param_to_dim_input["y"]] if "y" in self.param_to_dim_input.keys() else None
        input_z = _data_input[self.param_to_dim_input["z"]] if "z" in self.param_to_dim_input.keys() else None
        data_indices[0], data_indices[1] = self.__div_dataset(x = input_x, y = input_y, z = input_z)
        data_input = {}
        for num in self.iter_nums:
            data_input[num] = _data_input[:, data_indices[num]]

        for key, dim in self.param_to_dim_input.items():
            if key in ["ScaleFactor"]:
                continue
            for num in self.iter_nums:
                data_input[num][dim, ...] = self.Norm_input[key][num].run_predict(data_input[num][dim, ...])

        data_predict = {}
        for num in self.iter_nums:
            batch_input = None
            if data_input[num].shape[1] > self.batch_max:
                batch_input = np.arange(0, data_input[num].shape[1], self.batch_max).tolist()
                if batch_input[-1] != data_input[num].shape[1]:
                    batch_input.append(data_input[num].shape[1])

            if LP.use_cupy:
                ##use cupy
                if batch_input is None:
                    data_predict[num] = self.model[num].predict(cupy.asarray(data_input[num]),
                                                                is_training = False)
                else:
                    data_predict[num] = None
                    for j in range(len(batch_input) - 1):
                        batch_data_input = cupy.asarray(data_input[num][:, batch_input[j]:batch_input[j+1]])
                        predict = self.model[num].predict(batch_data_input,
                                                          is_training = False)
                        predict = cupy.asnumpy(predict)
                        if data_predict[num] is None:
                            data_predict[num] = predict
                        else:
                            data_predict[num] = np.concatenate([data_predict[num], predict], axis = 1)
                data_predict[num] = cupy.asnumpy(data_predict[num])

            else:
                ##not use cupy
                if batch_input is None:
                    data_predict[num] = self.model[num].predict(data_input[num],
                                                                is_training = False)
                else:
                    data_predict[num] = None
                    for j in range(len(batch_input) - 1):
                        batch_data_input = data_input[num][:, batch_input[j]:batch_input[j+1]]
                        predict = self.model[num].predict(batch_data_input,
                                                          is_training = False)
                        if data_predict[num] is None:
                            data_predict[num] = predict
                        else:
                            data_predict[num] = np.concatenate([data_predict[num], predict], axis = 1)

            for key, dim in self.param_to_dim_output.items():
                data_predict[num][dim, ...] = self.Norm_output[key][num].inv_run_predict(data_predict[num][dim, ...])

        rowsize = 0
        for num in self.iter_nums:
            rowsize += data_predict[num].shape[1]

        if not len(self.iter_nums) == 1:
            return_predict = np.empty((data_predict[0].shape[0], rowsize, data_predict[0].shape[2]))
            for num in self.iter_nums:
                return_predict[:, data_indices[num]] = data_predict[num]
        else:
            return_predict = data_predict[self.iter_nums[0]]

        return return_predict


    def plot_figures(self, save_dir, save_fig_type,
                     fontsize = 26, labelsize = 15,
                     length_major = 20, length_minor = 13,
                     linewidth = 2.5, figsize = (8, 5)):

        ##plot loss function
        for num in self.iter_nums:
            plt_label = "_under_threshold" if num == 0 else "_upper_threthold"
            fig = plt.figure(figsize = figsize)
            ax_loss = fig.add_subplot(111)
            for p_key in self.param_to_dim_output.keys():
                epochs = np.arange(len(self.loss_val[p_key + str(num)])) * self.split_epoch
                ax_loss.plot(epochs[1:], self.loss_val[p_key + str(num)][1:], label = "Loss Function", linewidth = linewidth)
                np.savetxt("{0}data_loss_{1}{2}.csv".format(save_dir, p_key, plt_label),
                           self.loss_val[p_key + str(num)],
                           delimiter = ",")
            ax_loss.set_yscale("log")
            ax_loss.set_xlabel("Epoch", fontsize = fontsize)
            ax_loss.set_ylabel("Loss Function", fontsize = fontsize)
            ax_loss.legend(loc = "best", fontsize = int(fontsize * 0.6))
            ax_loss.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
            ax_loss.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
            plt.tight_layout()
            plt.savefig("{}fig_loss{}".format(save_dir, save_fig_type))

        ##plot accuracy
        for p_key in self.param_to_dim_output.keys():
            for num in self.iter_nums:
                plt_label = "_under_threshold" if num == 0 else "_upper_threshold"
                epochs = np.arange(len(self.train_acc[p_key + str(num)])) * self.split_epoch
                fig = plt.figure(figsize = figsize)
                ax_acc = fig.add_subplot(111)
                ax_acc.plot(epochs[1:], self.train_acc[p_key + str(num)][1:], label = "Training Dataset", linewidth = linewidth, color = "orange")
                if self.is_acc_test:
                    ax_acc.plot(epochs[1:], self.test_acc[p_key + str(num)][1:], label = "Testing Dataset", linewidth = linewidth, color = "blue")
                ax_acc.set_yscale("log")
                ax_acc.set_xlabel("Epoch", fontsize = fontsize)
                ax_acc.set_ylabel("Accuracy", fontsize = fontsize)
                ax_acc.legend(loc = "best", fontsize = int(fontsize * 0.6))
                ax_acc.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
                ax_acc.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
                plt.tight_layout()
                plt.savefig("{0}fig_acc_{1}{2}{3}".format(save_dir, p_key, plt_label, save_fig_type))
                np.savetxt("{0}data_acc_train_{1}{2}.csv".format(save_dir, p_key, plt_label), self.train_acc[p_key + str(num)], delimiter = ",")
                if self.is_acc_test:
                    np.savetxt("{0}data_acc_test_{1}{2}.csv".format(save_dir, p_key, plt_label), self.test_acc[p_key + str(num)], delimiter = ",")
