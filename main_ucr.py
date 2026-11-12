# Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I Webb
#
# MultiRocket: Multiple pooling operators and transformations for fast and effective time series classification
# https://arxiv.org/abs/2102.00457
import argparse
import os
import platform
import socket
import time
from datetime import datetime

import numba
import numpy as np
import pandas as pd
import psutil
import pytz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from multirocket.multirocket import MultiRocket
from utils.data_loader import read_univariate_ucr, non_109_datasets
from utils.tools import create_directory

pd.set_option('display.max_columns', 500)

itr = 0
num_features = 10000
save = True
num_threads = 0

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datapath", type=str, required=False, default="C:/Users/changt/workspace/Dataset/UCRArchive_2018/")
parser.add_argument("-p", "--problem", type=str, required=False, default="ACSF1")
parser.add_argument("-i", "--iter", type=int, required=False, default=0)
parser.add_argument("-n", "--num_features", type=int, required=False, default=50000)
parser.add_argument("-t", "--num_threads", type=int, required=False, default=-1)
parser.add_argument("-s", "--save", type=bool, required=False, default=True)
parser.add_argument("-v", "--verbose", type=int, required=False, default=2)
parser.add_argument("--pooling", default="PPV+LSPV+MPV+MIPV",
                    help="Operators to retain; num_features keeps the original four-operator kernel budget")
parser.add_argument("--classifier", choices=("ridge", "fast_ridge", "auto_ridge", "logistic"), default="ridge")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--memory-efficient", action="store_true")
parser.add_argument("--prediction-batch-size", type=int, default=256)
parser.add_argument("--optimized-transform", action="store_true")
parser.add_argument("--fused-prediction", action="store_true")
parser.add_argument("--legacy-first-difference", action="store_true")
parser.add_argument("--ridge-feature-limit-mib", type=float, default=693)

arguments = parser.parse_args()

if __name__ == '__main__':
    data_path = arguments.datapath
    problem = arguments.problem
    num_features = arguments.num_features
    num_threads = arguments.num_threads
    itr = arguments.iter
    save = arguments.save
    verbose = arguments.verbose

    output_path = os.getcwd() + "/output/"
    classifier_name = "MultiRocket_{}".format(num_features)
    if (arguments.pooling != "PPV+LSPV+MPV+MIPV" or arguments.classifier != "ridge"
            or arguments.memory_efficient or arguments.seed is not None
            or arguments.optimized_transform or arguments.fused_prediction):
        classifier_name += "_{}_{}_mem{}_seed{}".format(
            arguments.pooling.replace(",", "+"), arguments.classifier,
            int(arguments.memory_efficient), arguments.seed)
        if arguments.optimized_transform or arguments.fused_prediction:
            classifier_name += "_buf{}_fused{}".format(
                int(arguments.optimized_transform), int(arguments.fused_prediction))

    if not arguments.legacy_first_difference:
        classifier_name += "_diff1fixed"
    if arguments.classifier == "auto_ridge":
        classifier_name += "_limit{}MiB".format(arguments.ridge_feature_limit_mib)

    data_folder = os.path.join(data_path, problem, "")

    if os.path.exists(data_folder):
        if num_threads > 0:
            numba.set_num_threads(num_threads)

        start = time.perf_counter()

        output_dir = "{}/multirocket/resample_{}/{}/{}/".format(
            output_path,
            itr,
            classifier_name,
            problem
        )
        if save:
            create_directory(output_dir)

        print("=======================================================================")
        print("Starting Experiments")
        print("=======================================================================")
        print("Data path: {}".format(data_path))
        print("Output Dir: {}".format(output_dir))
        print("Iteration: {}".format(itr))
        print("Problem: {}".format(problem))
        print("Number of Features: {}".format(num_features))

        # use tsv. Change the data loader for other file format
        train_file = data_folder + problem + "_TRAIN.tsv"
        test_file = data_folder + problem + "_TEST.tsv"

        print("Loading data")
        X_train, y_train = read_univariate_ucr(train_file, normalise=False)
        X_test, y_test = read_univariate_ucr(test_file, normalise=False)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        # returns ntc format, remove the last dimension
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

        if (itr > 0) and (problem not in non_109_datasets):
            all_data = np.vstack((X_train, X_test))
            all_labels = np.hstack((y_train, y_test))
            print(all_data.shape)

            all_indices = np.arange(len(all_data))
            training_indices = np.loadtxt("data/indices109/{}_INDICES_TRAIN.txt".format(problem),
                                          skiprows=itr,
                                          max_rows=1).astype(np.int32)
            test_indices = np.setdiff1d(all_indices, training_indices, assume_unique=True)

            X_train, y_train = all_data[training_indices, :], all_labels[training_indices]
            X_test, y_test = all_data[test_indices, :], all_labels[test_indices]

        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

        classifier = MultiRocket(
            num_features=num_features,
            classifier=arguments.classifier,
            verbose=verbose,
            pooling=arguments.pooling,
            random_state=arguments.seed,
            memory_efficient=arguments.memory_efficient,
            prediction_batch_size=arguments.prediction_batch_size,
            optimized_transform=arguments.optimized_transform,
            fused_prediction=arguments.fused_prediction,
            first_difference_fix=not arguments.legacy_first_difference,
            ridge_feature_limit_bytes=int(arguments.ridge_feature_limit_mib * 1024**2),
        )
        yhat_train = classifier.fit(
            X_train, y_train,
            predict_on_train=False
        )

        if yhat_train is not None:
            train_acc = accuracy_score(y_train, yhat_train)
        else:
            train_acc = -1

        yhat_test = classifier.predict(X_test)
        test_acc = accuracy_score(y_test, yhat_test)

        # get cpu information
        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        max_freq = cpu_freq.max
        min_freq = cpu_freq.min
        memory = np.round(psutil.virtual_memory().total / 1e9)

        df_metrics = pd.DataFrame(data=np.zeros((1, 21), dtype=float), index=[0],
                                  columns=['timestamp', 'itr', 'classifier',
                                           'num_features',
                                           'dataset',
                                           'train_acc', 'train_time',
                                           'test_acc', 'test_time',
                                           'generate_kernel_time',
                                           'apply_kernel_on_train_time',
                                           'apply_kernel_on_test_time',
                                           'train_transform_time',
                                           'test_transform_time',
                                           'machine', 'processor',
                                           'physical_cores',
                                           "logical_cores",
                                           'max_freq', 'min_freq', 'memory'])
        df_metrics["timestamp"] = datetime.utcnow().replace(tzinfo=pytz.utc).strftime("%Y-%m-%d %H:%M:%S")
        df_metrics["itr"] = itr
        df_metrics["classifier"] = classifier_name
        df_metrics["num_features"] = num_features
        df_metrics["dataset"] = problem
        df_metrics["train_acc"] = train_acc
        df_metrics["train_time"] = classifier.train_duration
        df_metrics["test_acc"] = test_acc
        df_metrics["test_time"] = classifier.test_duration
        df_metrics["generate_kernel_time"] = classifier.generate_kernel_duration
        df_metrics["apply_kernel_on_train_time"] = classifier.apply_kernel_on_train_duration
        df_metrics["apply_kernel_on_test_time"] = classifier.apply_kernel_on_test_duration
        df_metrics["train_transform_time"] = classifier.train_transforms_duration
        df_metrics["test_transform_time"] = classifier.test_transforms_duration
        df_metrics["machine"] = socket.gethostname()
        df_metrics["processor"] = platform.processor()
        df_metrics["physical_cores"] = physical_cores
        df_metrics["logical_cores"] = logical_cores
        df_metrics["max_freq"] = max_freq
        df_metrics["min_freq"] = min_freq
        df_metrics["memory"] = memory
        df_metrics["actual_num_features"] = classifier.n_features_out_
        df_metrics["pooling"] = "+".join(classifier.pooling)
        df_metrics["ridge_backend"] = arguments.classifier
        df_metrics["seed"] = arguments.seed
        df_metrics["memory_efficient"] = arguments.memory_efficient
        df_metrics["prediction_batch_size"] = arguments.prediction_batch_size
        df_metrics["optimized_transform"] = arguments.optimized_transform
        df_metrics["fused_prediction"] = arguments.fused_prediction
        df_metrics["first_difference_fix"] = not arguments.legacy_first_difference
        df_metrics["selected_ridge_backend"] = classifier.selected_ridge_backend_
        df_metrics["ridge_feature_limit_bytes"] = classifier.ridge_feature_limit_bytes
        df_metrics["fit_total_time"] = classifier.fit_total_duration
        df_metrics["predict_total_time"] = classifier.predict_total_duration
        df_metrics["train_feature_bytes"] = classifier.train_feature_bytes_
        df_metrics["max_prediction_feature_bytes"] = classifier.max_prediction_feature_bytes_
        if arguments.classifier in ("ridge", "fast_ridge", "auto_ridge"):
            df_metrics["alpha"] = classifier.classifier[-1].alpha_

        print(df_metrics)
        if save:
            df_metrics.to_csv(output_dir + 'results.csv', index=False)
    else:
        print(f"Error: Data folder does not exist: {data_folder}")
        print("Please check that the dataset path provided with -d/--datapath is valid.")

