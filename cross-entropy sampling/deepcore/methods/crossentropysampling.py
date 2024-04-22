import random
from collections import defaultdict

import keras.backend as K
import numpy as np

from .coresetmethod import CoresetMethod


class CrossEntropySampling(CoresetMethod):
    def __init__(self, dst_train, args, fraction, random_seed, y_test,model):
        super().__init__(dst_train, args, fraction, random_seed)
        self.n_train = len(dst_train)
        self.y_test=y_test
        self.model=model
        np.random.seed(random_seed)

    def build_neuron_tables(self,model, x_test, divide,test_output):
        total_num = x_test.shape[0]
        # init dict and its input
        neuron_interval = defaultdict(np.array)
        neuron_proba = defaultdict(np.array)
        input_tensor = model.input
        layer = model.layers[-3]
        output = test_output
        lower_bound = np.min(output, axis=0)
        upper_bound = np.max(output, axis=0)
        for index in range(output.shape[-1]):
            interval = np.linspace(
                lower_bound[index], upper_bound[index], divide)
            neuron_interval[(layer.name, index)] = interval
            neuron_proba[(layer.name, index)] = self.output_to_interval(
                output[:, index], interval) / total_num
        return neuron_interval, neuron_proba
    def build_testoutput(self,model, x_test):
        input_tensor = model.input
        layer = model.layers[-3]
        # get this layer's output
        output = layer.output
        output_fun = K.function([input_tensor], [output])
        output = output_fun([x_test])[0]
        output = output.reshape(output.shape[0], -1)
        test_output = output
        return test_output
    def neuron_entropy(self,model, neuron_interval, neuron_proba, sample_index,test_output):
        total_num = sample_index.shape[0]
        if(total_num == 0):
            return -1e3
        neuron_entropy = []
        layer = model.layers[-3]
        output = test_output
        output = output[sample_index, :]
        # get lower and upper bound of neuron output
        # lower_bound = np.min(output, axis=0)
        # upper_bound = np.max(output, axis=0)
        for index in range(output.shape[-1]):
            # compute interval
            interval = neuron_interval[(layer.name, index)]
            bench_proba = neuron_proba[(layer.name, index)]
            test_proba = self.output_to_interval(
                output[:, index], interval) / total_num
            test_proba = np.clip(test_proba, 1e-10, 1 - 1e-10)
            log_proba = np.log(test_proba)
            temp_proba = bench_proba.copy()
            temp_proba[temp_proba < (.5 / total_num)] = 0
            entropy = np.sum(log_proba * temp_proba)
            neuron_entropy.append(entropy)
        return np.array(neuron_entropy)
    def coverage(self,neuron_entropy):
        return np.mean(neuron_entropy)
    def output_to_interval(self,output, interval):
        num = []
        for i in range(interval.shape[0] - 1):
            num.append(np.sum(np.logical_and(
                output > interval[i], output < interval[i + 1])))
        return np.array(num)
    def selectsample(self,model, x_test, y_test, delta, iterate,neuron_interval,neuron_proba,test_output):
        test = x_test
        batch = delta
        max_iter=int(len(self.dst_train)*self.fraction)
        print(max_iter,)
        arr = np.random.permutation(test.shape[0])
        max_index0 = arr[0:max_iter]
        acc_list1 = []
        cov_select = []
        arr = np.random.permutation(test.shape[0])
        max_coverage = -1e3
        min_coverage = 0
        max_index = -1
        min_index = -1
        e = self.neuron_entropy(model, neuron_interval,
                        neuron_proba, max_index0,test_output)
        cov = self.coverage(e)
        max_coverage = cov

        temp_cov = []
        index_list = []
        
        # select
        for j in range(max_iter):
            arr = np.random.permutation(test.shape[0])
            start = int(np.random.uniform(0, test.shape[0] - batch))
            temp_index = np.append(max_index0, arr[start:start + batch])
            index_list.append(arr[start:start + batch])
            e = self.neuron_entropy(model, neuron_interval,
                            neuron_proba, temp_index,test_output)
            new_coverage = self.coverage(e)
            temp_cov.append(new_coverage)

        max_coverage = np.max(temp_cov)
        cov_index = np.argmax(temp_cov)
        max_index = index_list[cov_index]
        if(max_coverage <= cov):
            start = int(np.random.uniform(0, test.shape[0] - batch))
            max_index = arr[start:start + batch]
        max_index0 = np.append(max_index0, max_index)
        label = y_test[max_index0]
        orig_sample = x_test[max_index0]
        orig_sample = orig_sample.reshape(-1, 32, 32, 3)
        pred = np.argmax(model.predict(orig_sample), axis=1)
        acc1 = np.sum(pred == label) / orig_sample.shape[0]
        max_index0_sample = np.random.choice(max_index0, max_iter)
        return acc1,max_index0_sample
    def experiments(self,delta, iterate,neuron_interval,neuron_proba,test_output):
        pred = np.argmax(self.model.predict(self.dst_train), axis=1)
        true_acc = np.sum(pred == self.y_test) / self.dst_train.shape[0]
        print("The final acc is {!s}".format(true_acc))
        acc_list1,indices = self.selectsample(
            self.model,self.dst_train, self.y_test, delta, iterate,neuron_interval,neuron_proba,test_output)
        return np.array(acc_list1), np.array(indices)
    def select(self):
        print("in function")
        test_output = self.build_testoutput(self.model, self.dst_train)
        neuron_interval, neuron_proba = self.build_neuron_tables(
            self.model, self.dst_train,5,test_output)
        acc,indices=self.experiments(5, 30,neuron_interval,neuron_proba,test_output)
        print("acc:",acc)
        return {"indices": indices}
