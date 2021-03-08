#!/usr/bin/env python
# coding: utf-8
print("[INFO] Import Library...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import shutil

import padasip as pa
from padasip.filters import AdaptiveFilter

def preprocessing_AFDB(record, start=1, stop=None, sep=",", fs=250, sample_size=6):
    dataset_dir = "dataset/AFDB record_%s/" % record
    csv_filenames = []
    for filename in os.listdir(dataset_dir) :
        if filename.find(".csv") > -1:
            csv_filenames.append(filename)
    print("[INFO] detected CSV file :", csv_filenames)
            
    print("[INFO] Read annotation file...")
    file = open(dataset_dir + 'annotation.txt',"r") 
    annotations = file.readlines()
    file.close()

    label_idx = []
    for item in annotations[start:stop] :
        item_split = item.split()
        label_idx.append([item_split[0].replace("[", "").replace("]", ""), item_split[-1].replace("(", "")])

    print("[INFO] Read CSV...")
    # - Read & formatting ECG data
    def read_csv_to_df(filename, folder, sep=sep):
        df = pd.read_csv(folder + filename, sep=sep)
        df = df.iloc[:, 0:2]
        print("[INFO] finish read file - %s" % filename)

        #df = df.drop(0) 
        df.columns = ['Time', 'ECG']

        #df['ECG'] = df['ECG'].str.replace(';', '')
        df['ECG'] = pd.to_numeric(df['ECG'])

        # peak reduction
        df[df['ECG'] > 2] = 2
        df[df['ECG'] < -2] = -2
        print("[INFO] finish data cleansing - %s" % filename)

        df["Time"] = df['Time'].str.replace("[", "")
        df["Time"] = df['Time'].str.replace("]", "")
        df["Time"] = df['Time'].str.replace("'", "")

        df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
        print("[INFO] finish time cleansing -  %s" % filename)

        df.set_index("Time", inplace=True)
        return df


    # - concate datafarame
    list_df_ecg = []
    for name in csv_filenames:
        df = read_csv_to_df(name, dataset_dir)
        list_df_ecg.append(df)

    df_ecg = pd.concat(list_df_ecg)
    label_idx.append([str(df_ecg.index[-1].time()), ''])

    # - Split Normal (N) and AFIB data
    N_range = []
    AFIB_range = []

    for i in range(len(label_idx) - 1):
        tm_str = label_idx[i][0]
        next_tm_str = label_idx[i + 1][0]
        tm = pd.to_datetime(tm_str)
        next_tm = pd.to_datetime(next_tm_str)

        if label_idx[i][1] == 'N' :
            N_range.append([tm, next_tm])
        else :
            AFIB_range.append([tm, next_tm])
    
    if not os.path.exists("dataset_split_per_class"):
        os.mkdir("dataset_split_per_class")
    
    N = []
    for ix, nr in enumerate(N_range) :
        result = df_ecg.between_time(nr[0].time(), nr[1].time())
        result.to_csv("dataset_split_per_class/%s_%s_%s_%s.csv" % 
                      ('N', record, 'ECG1', ix))
        N.append(result)

    AFIB = []
    for ix, ar in enumerate(AFIB_range) :
        result = df_ecg.between_time(ar[0].time(), ar[1].time())
        result.to_csv("dataset_split_per_class/%s_%s_%s_%s.csv" % 
                      ('AF', record, 'ECG1', ix))
        AFIB.append(result)


    print("[INFO] Split per-6s & apply Baseline Wander Removal")
    # - split each N & AFIB dataframe to 16s sequence and apply Baseline Removal 
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from datetime import timedelta


    def baseline_als(y, lam=10000, p=0.05, n_iter=10):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(n_iter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z

    def perdelta(start, end, delta):
        curr = start
        while curr < end:
            yield curr
            curr += delta

    time_interval_N = []
    for N_item in N:
        if len(N_item) > 0:
            intr = [time_result for time_result in perdelta(N_item.index[0], N_item.index[-1], timedelta(seconds=sample_size))]
            time_interval_N.append(intr)


    time_interval_AFIB = []
    for AFIB_item in AFIB:
        if len(AFIB_item) > 0:
            intr = [time_result for time_result in perdelta(AFIB_item.index[0], AFIB_item.index[-1], timedelta(seconds=sample_size))]
            time_interval_AFIB.append(intr)

    ECG_ALS = []
    ECG_ALS_label = []

    for time_interval in time_interval_N :
        for time_intv in list(zip(time_interval, time_interval[1:])):
            X = df_ecg.between_time(time_intv[0].time(), time_intv[1].time())
            if len(X) > 0 and (X.index[-1] - X.index[0]).total_seconds() >= sample_size :
                X_val = X.values[:,0]
                if len(X_val) > 0 :
                    ALS = X_val - baseline_als(X_val)
                    ECG_ALS.append(np.array(ALS))
                    ECG_ALS_label.append('N')

    for time_interval in time_interval_AFIB :
        for time_intv in list(zip(time_interval, time_interval[1:])):
            X = df_ecg.between_time(time_intv[0].time(), time_intv[1].time())
            if len(X) > 0 and (X.index[-1] - X.index[0]).total_seconds() >= sample_size :
                X_val = X.values[:,0]
                if len(X_val) > 0 :
                    ALS = X_val - baseline_als(X_val)
                    ECG_ALS.append(np.array(ALS))
                    ECG_ALS_label.append('AF')


    print("[INFO] Signal Normalization...")
    # - Signal normalization from -1 to 1
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

    scaler = MaxAbsScaler()
    ECG_ALS_Norm = []

    for als in ECG_ALS :
        als = np.expand_dims(als, 1)
        scaler = scaler.fit(als)

        als_norm = scaler.transform(als) 
        ECG_ALS_Norm.append(als_norm)


#     print("[INFO] R-R peak detection & split ...")
#     # - QRS Detection
#     from ecgdetectors import Detectors

#     detectors = Detectors(fs)

#     # - Split each 16s to 1.2 x R-R sequence
#     # - Padding the sequence with zero for length 300 point

#     ECG_split = []
#     ECG_split_label = []
#     for i in range(len(ECG_ALS_Norm)) :
#         data = np.array(ECG_ALS_Norm[i])
#         if len(data) > 0:
#             r_peaks = []
#             try :
#                 r_peaks = detectors.christov_detector(data)
#             except:
#                 print("cannot find R peaks in ALS Norm, idx %d" % i)
#             RRs = np.diff(r_peaks)
#             RRs_med = np.median(RRs)
#             if not np.isnan(RRs_med) and RRs_med > 0 and len(r_peaks) > 0:
#                 for rp in r_peaks[:-1] :
#                     split = data[:,0][rp : rp + int(RRs_med * 1.2)] 
#                     pad = np.zeros(300)
#                     n = len(split) if len(split) <= 300 else 300
#                     pad[0:n] = split[0:n]
#                     ECG_split.append(pad)
#                     ECG_split_label.append(ECG_ALS_label[i])

    ECG_split = []
    ECG_split_label = []
    for i in range(len(ECG_ALS_Norm)) :
        data = np.array(ECG_ALS_Norm[i])[:,0]
        pad = np.zeros(sample_size*fs)
        n = len(data) if len(data) <= sample_size*fs else sample_size*fs
        pad[0:n] = data[0:n]
        ECG_split.append(pad)
        ECG_split_label.append(ECG_ALS_label[i])
        
    print("[INFO] Save preprocessed data to CSV file for record %s..." % record)
    data = []
    for i in range(len(ECG_split)):
        x = list(ECG_split[i])
        x.append(ECG_split_label[i])
        data.append(x)

    ECG = pd.DataFrame(data)
    ECG.to_csv("dataset/AFDB_%s_sequence_%s_pt.csv" % (record, fs*sample_size), index=False, header=False)

    print("-------------------------- *** --------------------------\n\n")

def preprocessing_NSRDB(record, fs = 128, sample_size=6):
    dataset_dir = "dataset/NSRDB/%s/" % record 
    
    csv_filenames = []
    for filename in os.listdir(dataset_dir) :
        if filename.find(".csv") > -1:
            csv_filenames.append(filename)
    print("[INFO] detected CSV file :", csv_filenames)
            
    print("[INFO] Read CSV...")
    def read_csv_to_df(filename, folder, sep=","):
        df = pd.read_csv(folder + filename, sep=sep)
        df = df.iloc[:, 0:2]
        print("[INFO] finish read file - %s" % filename)

        #df = df.drop(0) 
        df.columns = ['Time', 'ECG']

        #df['ECG'] = df['ECG'].str.replace(';', '')
        df['ECG'] = pd.to_numeric(df['ECG'])

        # peak reduction
        df[df['ECG'] > 2] = 2
        df[df['ECG'] < -2] = -2
        print("[INFO] finish data cleansing - %s" % filename)

        df["Time"] = df['Time'].str.replace("[", "")
        df["Time"] = df['Time'].str.replace("]", "")
        df["Time"] = df['Time'].str.replace("'", "")

        df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
        print("[INFO] finish time cleansing -  %s" % filename)

        df.set_index("Time", inplace=True)
        return df
    
    list_df_ecg = []
    for name in csv_filenames:
        df = read_csv_to_df(name, dataset_dir)
        list_df_ecg.append(df)

    df_ecg = pd.concat(list_df_ecg)

    print("[INFO] Split per-6s & apply Baseline Wander Removal")
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from datetime import timedelta
    
    def baseline_als(y, lam=10000, p=0.05, n_iter=10):
        L = len(y)
        D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
        w = np.ones(L)
        for i in range(n_iter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return z
    
    def perdelta(start, end, delta):
        curr = start
        while curr < end:
            yield curr
            curr += delta
            
    time_interval = []
    if len(df_ecg) > 0:
        intr = [time_result for time_result in perdelta(df_ecg.index[0], df_ecg.index[-1], timedelta(seconds=sample_size))]
        time_interval.append(intr)
        
    ECG_ALS = []
    ECG_ALS_label = []

    for tm_int in time_interval :
        for time_intv in list(zip(tm_int, tm_int[1:])):
            X = df_ecg.between_time(time_intv[0].time(), time_intv[1].time())
            if len(X) > 0 and (X.index[-1] - X.index[0]).total_seconds() >= sample_size :
                X_val = X.values[:,0]
                if len(X_val) > 0 :
                    ALS = X_val - baseline_als(X_val)
                    ECG_ALS.append(np.array(ALS))
                    ECG_ALS_label.append('N')
      
    print("[INFO] Signal Normalization...")
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
    scaler = MaxAbsScaler()
    ECG_ALS_Norm = []

    for als in ECG_ALS :
        als = np.expand_dims(als, 1)
        scaler = scaler.fit(als)

        als_norm = scaler.transform(als) 
        ECG_ALS_Norm.append(als_norm)
        
    print("[INFO] upsampling signal to 250Hz ...")
    def upsampling_twice(data):
        # upsampling interpolation
        result = np.zeros(2*len(data)-1)
        result[0::2] = data
        result[1::2] = (data[1:] + data[:-1]) / 2
        return result
    
    new_fs = 250 # Hz 
    ECG_ALS_Norm_Up = []
    for data in ECG_ALS_Norm :
        data = np.array(data[:,0])
        data = upsampling_twice(data) 
        ECG_ALS_Norm_Up.append(data)
        
#     print("[INFO] R-R peak detection & split ...")
#     from ecgdetectors import Detectors
#     detectors = Detectors(new_fs)
    
#     ECG_split = []
#     ECG_split_label = []
#     for i in range(len(ECG_ALS_Norm_Up)) :
#         data = np.array(ECG_ALS_Norm_Up[i])
#         if len(data) > 0:
#             r_peaks = []
#             try :
#                 r_peaks = detectors.christov_detector(data)
#             except:
#                 print("cannot find R peaks in ALS Norm, idx %d" % i)
#             RRs = np.diff(r_peaks)
#             RRs_med = np.median(RRs)
#             if not np.isnan(RRs_med) and RRs_med > 0:
#                 for rp in r_peaks :
#                     split = data[rp : rp + int(RRs_med * 1.2)] 
#                     pad = np.zeros(300)
#                     n = len(split) if len(split) <= 300 else 300
#                     pad[0:n] = split[0:n]
#                     ECG_split.append(pad)
#                     ECG_split_label.append(ECG_ALS_label[i])

    ECG_split = []
    ECG_split_label = []
    for i in range(len(ECG_ALS_Norm_Up)) :
        data = np.array(ECG_ALS_Norm_Up[i])
        pad = np.zeros(sample_size*new_fs)
        n = len(data) if len(data) <= sample_size*new_fs else sample_size*new_fs
        pad[0:n] = data[0:n]
        ECG_split.append(pad)
        ECG_split_label.append(ECG_ALS_label[i])
        
    print("[INFO] Save preprocessed data to CSV file for record %s..." % record)
    data = []
    for i in range(len(ECG_split)):
        x = list(ECG_split[i])
        x.append(ECG_split_label[i])
        data.append(x)

    ECG = pd.DataFrame(data)
    ECG.to_csv("dataset/NSRDB_%s_sequence_%s_pt.csv" % (record, new_fs*sample_size), index=False, header=False)
    print("-------------------------- *** --------------------------\n\n")
    
def balancing_dataset(record, n_samples, fs=250, sample_size=6):  
    import pandas as pd
    import numpy as np 
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample

    print("[INFO] read preprocessed record :  %s" % record)
    dataset_folder = "dataset/"
    filename = dataset_folder + 'AFDB_%s_sequence_%s_pt.csv' % (record, fs*sample_size)
    ecg_df = pd.read_csv(filename, header=None)
    print(filename)

    X = ecg_df.iloc[:,:fs*sample_size].values
    y = ecg_df.iloc[:,fs*sample_size].values

    le = LabelEncoder()
    le.fit(['AF', 'N'])
    labels = le.classes_
    print(labels)
    y = le.transform(y)

    print("[INFO] split data...")
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.33, random_state=42)

    train_df = pd.DataFrame(np.hstack((X_train, np.expand_dims(y_train, 1))))
    test_df = pd.DataFrame(np.hstack((X_test, np.expand_dims(y_test, 1))))

    train_df[fs*sample_size]=train_df[fs*sample_size].astype(int)
    equilibre=train_df[fs*sample_size].value_counts()
    
    if n_samples != None :
        print("[INFO] balancing data...")
        # sampling and resampling dataset
        random_states = [42, 123]
        dfs = []
        for i in range(len(equilibre)):
            dfs.append(train_df[train_df[fs*sample_size]==i])
            if(equilibre[i] > n_samples) :
                dfs[i]=dfs[i].sample(n=n_samples ,random_state=random_states[i])
            else :
                dfs[i]=resample(dfs[i],replace=True,n_samples=n_samples,random_state=random_states[i])
        train_df=pd.concat(dfs)
    else :
        print("[INFO] `n_samples` for record %s is None, the data will be left unbalanced..." % record)
        df = pd.DataFrame(np.hstack((X, np.expand_dims(y, 1))))
        equilibre=df[fs*sample_size].value_counts()
        print(equilibre)
        
    print("[INFO] save balanced data...")
    train_df.to_csv(dataset_folder + "train_AFDB_%s_balanced.csv" % record, header=None, index=None)
    test_df.to_csv(dataset_folder + "test_AFDB_%s.csv" % record, header=None, index=None)
    print("-------------------------- *** --------------------------\n\n")

def merging_dataset(n_samples = 4000, fs=250, sample_size=6):
    dataset_folder = 'dataset/'
    filenames = []
    for fn in os.listdir(dataset_folder):
        if (fn.find("_%d_pt" % (fs*sample_size)) > -1 and fn.find("_NSRDB_") > -1) or fn.find("_AFDB_") > -1:
            filenames.append(fn)

    train_dfs = []
    test_dfs = []
    normal_dfs = []
    print("[INFO] read all balanced dataset...")
    for name in filenames :
        if name.find('train_') > -1:
            train_df = pd.read_csv(dataset_folder + name, header=None)
            train_dfs.append(train_df)
        if name.find('test_') > -1:
            test_df = pd.read_csv(dataset_folder + name, header=None)
            test_dfs.append(test_df)
        if name.find('_NSRDB_') > -1:
            normal_df = pd.read_csv(dataset_folder + name, header=None)
            normal_dfs.append(normal_df)
        
    print("[INFO] merging all dataset...")
    train_df_all = pd.concat(train_dfs, ignore_index=True)
    test_df_all = pd.concat(test_dfs, ignore_index=True)
    normal_df_all = pd.concat(normal_dfs, ignore_index=True)
    
    train_df_AF = train_df_all[train_df_all[fs*sample_size] == 0]
    test_df_AF = test_df_all[test_df_all[fs*sample_size] == 0]
    normal_df_all[fs*sample_size] = 1
    
    df_AF_N = pd.concat([train_df_AF, test_df_AF, normal_df_all])
    
    print("[INFO] balancing after merging..")
    
    df_AF_N[fs*sample_size]=df_AF_N[fs*sample_size].astype(int)
    equilibre=df_AF_N[fs*sample_size].value_counts()
    print("[INFO] sample count before balancing...")
    print(equilibre)
    
    from sklearn.utils import resample
    random_states = [123, 124]
    dfs = []
    for i in range(len(equilibre)):
        dfs.append(df_AF_N[df_AF_N[fs*sample_size]==i])
        dfs[i]=resample(dfs[i],replace=True,n_samples=n_samples,random_state=random_states[i])
    df_AF_N_balanced =pd.concat(dfs)

    df_AF_N_balanced[fs*sample_size]=df_AF_N_balanced[fs*sample_size].astype(int)
    equilibre=df_AF_N_balanced[fs*sample_size].value_counts()
    print("[INFO] sample count after balancing...")
    print(equilibre)
    
    print("[INFO] split dataset...")
    from sklearn.model_selection import train_test_split
    
    print("[INFO] save final dataset ...")
    y = df_AF_N_balanced.iloc[:, fs*sample_size].values
    X = df_AF_N_balanced.iloc[:, :fs*sample_size].values
    
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.15, random_state=42)
    
    train_df_all = pd.DataFrame(np.hstack((X_train, np.expand_dims(y_train, 1))))
    test_df_all = pd.DataFrame(np.hstack((X_test, np.expand_dims(y_test, 1))))
    train_df_all.to_csv(dataset_folder + "train_all.csv", index=None, header=None)
    test_df_all.to_csv(dataset_folder + "test_all.csv", index=None, header=None)
    print("-------------------------- *** --------------------------\n\n")
       
class adapfilt_modeler():
    def __init__(self, model='lms', mu=0.1):
        self._model_name = model
        self._model = AdaptiveFilter(model=model, n=1, mu=mu, w="random")
        #self._output = []

    def fit(self, noised_signals, signals):
        print('------------------- %s model training ----------------------' % self._model_name.upper())
        for i, (s, ns) in enumerate(zip(signals, noised_signals)):
            x = np.reshape(s, (s.shape[0], 1))
            d = np.reshape(ns, (ns.shape[0], 1))
            y, e, w = self._model.run(d, x) 
            #self._output.append([y, e, w, d, x])

            if i % int(signals.shape[0]/10) == 0 :
                #snr = calc_snr(y, d)
                mse = pa.misc.MSE(e)
                rmse = pa.misc.RMSE(e)
                print("sample %d \t : MSE %.6f, RMSE %.6f" % (i, mse, rmse))

    def fit_transform(self, noised_signals, signals):
        self.fit(noised_signals, signals)

        return self.transform(noised_signals)

    def transform(self, signals):
        vec = np.vectorize(self._model.predict)   
        result = vec(signals)

        return result
        
def denoising(fs=250, sample_size=6):
    dataset_folder = "dataset/"
    train_df = pd.read_csv(dataset_folder + "train_all.csv", header=None)
    test_df = pd.read_csv(dataset_folder + "test_all.csv" , header=None)
    
    def add_AWGN_noise(signal, target_noise_db = -30):
        mean_noise = 0
        target_noise_watts = 10 ** (target_noise_db / 10)
        sigma = np.sqrt(target_noise_watts)
        noise = np.random.normal(mean_noise, sigma, len(signal))
        return (signal+noise)
    
    print("[INFO] load final dataset...")
    X_train=train_df.iloc[:,:fs*sample_size].values
    X_test=test_df.iloc[:,:fs*sample_size].values
    y_train=train_df.iloc[:,fs*sample_size].values
    y_test=test_df.iloc[:,fs*sample_size].values
    
    print("[INFO] add noise to dataset...")
    X_train_noised = train_df.iloc[:,:fs*sample_size].apply(add_AWGN_noise, axis=1).values
    X_test_noised = test_df.iloc[:,:fs*sample_size].apply(add_AWGN_noise, axis=1).values
    
    import pickle
    def save_adapfilt_model(model, filename, path=""): 
        with open(os.path.join(path, filename), 'wb') as out_name:
            pickle.dump(model, out_name, pickle.HIGHEST_PROTOCOL)

    def read_adapfilt_model(filename, path=""):
        with open(os.path.join(path, filename), 'rb') as in_name:
            model = pickle.load(in_name)
            return model
        
    def calc_snr(signal, noised_signal):
        idx = np.max(np.nonzero(signal))
        signal = signal[:idx]
        noised_signal = noised_signal[:idx]
        noise = noised_signal - signal
        std_noise = np.std(noise)
        signal_avg = np.mean(signal)
        SNR  = signal_avg/std_noise if signal_avg > 0 else 1
        SNR = 10*np.log(SNR)
        return SNR
    
    print("\n----- SNR Noised Signal -----")
    for i, (s, ns) in enumerate(zip(X_train, X_train_noised)):
        if i % int(X_train.shape[0]/10) == 0 :
            snr_db = calc_snr(s, ns)
            print("sample %d \t : SNR %.2fdb" % (i, snr_db))
        
    print("\n[INFO] training denoising model...")
    adapfilt_lms = adapfilt_modeler(model='lms', mu=0.1)
    adapfilt_lms.fit(X_train_noised, X_train)
    X_train_denoised_lms = adapfilt_lms.transform(X_train)
    print("\n----- SNR Denoised Signal LMS-----")
    for i, (s, ns) in enumerate(zip(X_train, X_train_denoised_lms)):
        if i % int(X_train.shape[0]/10) == 0 :
            snr_db = calc_snr(s, ns)
            print("sample %d \t : SNR %.3fdb" % (i, snr_db))
            
    adapfilt_nlms = adapfilt_modeler(model='nlms', mu=0.1)
    adapfilt_nlms.fit(X_train_noised, X_train)
    X_train_denoised_nlms = adapfilt_nlms.transform(X_train)
    print("\n----- SNR Denoised Signal NLMS-----")
    for i, (s, ns) in enumerate(zip(X_train, X_train_denoised_nlms)):
        if i % int(X_train.shape[0]/10) == 0 :
            snr_db = calc_snr(s, ns)
            print("sample %d \t : SNR %.3fdb" % (i, snr_db))
            
    adapfilt_rls = adapfilt_modeler(model='rls', mu=0.99)
    adapfilt_rls.fit(X_train_noised, X_train)
    X_train_denoised_rls = adapfilt_rls.transform(X_train)
    print("\n----- SNR Denoised Signal RLS-----")
    for i, (s, ns) in enumerate(zip(X_train, X_train_denoised_rls)):
        if i % int(X_train.shape[0]/10) == 0 :
            snr_db = calc_snr(s, ns)
            print("sample %d \t : SNR %.3fdb" % (i, snr_db))
            
    print("\n[INFO] save denoising model...")
    save_adapfilt_model(adapfilt_lms, "LMS_adapfilt_model_afdb.pkl")
    save_adapfilt_model(adapfilt_nlms, "NLMS_adapfilt_model_afdb.pkl")
    save_adapfilt_model(adapfilt_rls, "RLS_adapfilt_model_afdb.pkl")
    
    print("[INFO] load denoising model & test to test dataset...")
    model_lms = read_adapfilt_model("LMS_adapfilt_model_afdb.pkl")
    model_nlms = read_adapfilt_model("NLMS_adapfilt_model_afdb.pkl")
    model_rls = read_adapfilt_model("RLS_adapfilt_model_afdb.pkl")
    
    X_test_denoised_lms = adapfilt_lms.transform(X_test)
    X_test_denoised_nlms = adapfilt_nlms.transform(X_test)
    X_test_denoised_rls = adapfilt_rls.transform(X_test)
    
    print("[INFO] save denoised & noised final dataset...")
    train_lms_df = pd.DataFrame(np.hstack((X_train_denoised_lms, np.expand_dims(y_train, 1))))
    train_nlms_df = pd.DataFrame(np.hstack((X_train_denoised_nlms, np.expand_dims(y_train, 1))))
    train_rls_df = pd.DataFrame(np.hstack((X_train_denoised_rls, np.expand_dims(y_train, 1))))
    train_noised_df = pd.DataFrame(np.hstack((X_train_noised, np.expand_dims(y_train, 1))))

    test_lms_df = pd.DataFrame(np.hstack((X_test_denoised_lms, np.expand_dims(y_test, 1))))
    test_nlms_df = pd.DataFrame(np.hstack((X_test_denoised_nlms, np.expand_dims(y_test, 1))))
    test_rls_df = pd.DataFrame(np.hstack((X_test_denoised_rls, np.expand_dims(y_test, 1))))
    test_noised_df = pd.DataFrame(np.hstack((X_test_noised, np.expand_dims(y_test, 1))))
    
    train_lms_df.to_csv(dataset_folder + "train_all_lms.csv", index=None, header=None)
    train_nlms_df.to_csv(dataset_folder + "train_all_nlms.csv", index=None, header=None)
    train_rls_df.to_csv(dataset_folder + "train_all_rls.csv", index=None, header=None)
    train_noised_df.to_csv(dataset_folder + "train_all_noised.csv", index=None, header=None)

    test_lms_df.to_csv(dataset_folder + "test_all_lms.csv", index=None, header=None)
    test_nlms_df.to_csv(dataset_folder + "test_all_nlms.csv", index=None, header=None)
    test_rls_df.to_csv(dataset_folder + "test_all_rls.csv", index=None, header=None)
    test_noised_df.to_csv(dataset_folder + "test_all_noised.csv", index=None, header=None)
    print("-------------------------- *** --------------------------\n\n")
    
def classification(denoised = 'noised', cv_splits=5, EPOCHS = 16, BATCH_SIZE = 128, fs = 250, sample_size=6):    
    labels = ['AF', 'N']
    dataset_folder = 'dataset/'

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import confusion_matrix
    from sklearn.utils import class_weight
    from keras.utils.np_utils import to_categorical
    from sklearn.model_selection import StratifiedKFold

    print("[INFO] load final %s dataset ..." % denoised)
    if denoised == 'lms':
        train_df = pd.read_csv(dataset_folder + "train_all_lms.csv", header=None)
        test_df = pd.read_csv(dataset_folder + "test_all_lms.csv", header=None)

    elif denoised == 'nlms':
        train_df = pd.read_csv(dataset_folder + "train_all_nlms.csv", header=None)
        test_df = pd.read_csv(dataset_folder + "test_all_nlms.csv", header=None)

    elif denoised == 'rls':
        train_df = pd.read_csv(dataset_folder + "train_all_rls.csv", header=None)
        test_df = pd.read_csv(dataset_folder + "test_all_rls.csv", header=None)

    elif denoised == 'ori':
        train_df = pd.read_csv(dataset_folder + "train_all.csv", header=None)
        test_df = pd.read_csv(dataset_folder + "test_all.csv", header=None)

    elif denoised == 'noised':
        train_df = pd.read_csv(dataset_folder + "train_all_noised_-5db.csv", header=None)
        test_df = pd.read_csv(dataset_folder + "test_all_noised_-5db.csv", header=None)

    ecg_df = pd.concat([train_df, test_df])
    target_train = ecg_df[fs*sample_size]
    y = target_train
    X = ecg_df.iloc[:,:fs*sample_size].values
    
    
    # define traditional split
    #     X_train, X_test, y_train, y_test = train_test_split(
    #                                     X, y, test_size=0.15, random_state=42)
    
    # define Stratified K-Fold Cross Validation Split
    kf = StratifiedKFold(n_splits = cv_splits, random_state = 7, shuffle = True)
    

    print("\n\n")
    print("[INFO] ---------- Classification CNN ---------------")     
    print("[INFO] build model ...")
    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, MaxPool1D, Flatten, Dropout
    from keras.layers import Input
    from keras.models import Model
    from keras.layers.normalization import BatchNormalization
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    import keras
    
    def cnn_model(max_len):
    
        model = Sequential()

        model.add(Conv1D(filters=64,
                         kernel_size=5,
                         activation='relu',
                         input_shape=(max_len, 1)))
        model.add(BatchNormalization())
        model.add(MaxPool1D(pool_size=2,
                            strides=2,
                            padding='same'))

        model.add(Conv1D(filters=64,
                         kernel_size=3,
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool1D(pool_size=2,
                            strides=2,
                            padding='same'))

        model.add(Conv1D(filters=64,
                         kernel_size=3,
                         activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPool1D(pool_size=2,
                            strides=2,
                            padding='same'))

        # Fully Connected layer (FC)
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(128, 
                        activation='relu'))
        model.add(Dense(32, 
                        activation='relu'))
        model.add(Dense(2, 
                        activation='softmax'))

        model.summary()
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy',
                      metrics = ['accuracy'])

        return model
    
    def check_model(model_, x, y, x_val, y_val, epochs_, batch_size_, fold_var):
        callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                     ModelCheckpoint(filepath='best_model_cv%d.h5' % fold_var, 
                                    monitor='val_loss', save_best_only=True, mode='min')]

        hist = model_.fit(x, 
                          y,
                          epochs=epochs_,
                          callbacks=callbacks, 
                          batch_size=batch_size_,
                          shuffle=True,
                          validation_data=(x_val,y_val))
        model_.load_weights('best_model_cv%d.h5' % fold_var)
        return hist 
    
    fold_var = 1
    n_samples = len(y)
    for train_index, val_index in kf.split(np.zeros(n_samples), y):
        print("\n")
        print("[INFO] Train model... cv %d" % fold_var)
        print("\n")
        
        X_train = X[train_index] 
        X_test = X[val_index]
        y_ = to_categorical(y)
        y_train = y_[train_index]
        y_test = y_[val_index]

        X_train = X_train.reshape(len(X_train), X_train.shape[1],1)
        X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
        
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
            
        max_len = X_train.shape[1]  
        model = cnn_model(max_len)
        history=check_model(model, X_train,y_train,X_test,y_test, EPOCHS, BATCH_SIZE, fold_var)

        #model.save("CNN_Classification_model_%s.h5" % denoised)
        shutil.copy('best_model_cv%d.h5' % fold_var , "CNN_Classification_model_%s.h5" % denoised)
        pd.DataFrame.from_dict(history.history).to_csv('history_train_classif_cnn_denoising_%s_cv%d.csv' % 
                                                       (denoised, fold_var) ,index=False) 
            
        print("\n")
        print("[INFO] evaluate model - cv %d..." % fold_var) 
        print("\n")
        # predict test data
        y_pred=model.predict(X_test)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        print("Confusion Matrix - cv %d : \n" % fold_var, cnf_matrix)
        with open("confusion matrix - %s - cv%d.txt" % ('cnn - %s' % denoised, fold_var), 'w') as f:
            f.write(np.array2string(cnf_matrix, separator=', '))

        # print classification report
        cr = classification_report(y_test.argmax(axis=1), 
                                y_pred.argmax(axis=1), 
                                target_names=['AF', 'N'])
        print("Classification Report - cv %d: \n" % fold_var, cr)
        with open("classification report - %s - cv%d.txt" % ('cnn - %s' % denoised, fold_var), 'w') as f:
            f.write(cr)

        # clear session             
        keras.backend.clear_session()
        fold_var += 1
    print("-------------------------- *** --------------------------\n\n")
    
if __name__ == "__main__" :
    records = {
        "04015" : [1, None, None, ','], #8, 400
        "04043" : [1, None, None, ','], #16, 1000
        "04048" : [1, None, None, ','], #6, 900
        "04126" : [1, None, None, ','],
        "04746" : [1, None, None, ','],
        "04908" : [1, None, None, ','],
        "04936" : [1, None, None, ','], #2000
        "05091" : [1, None, None, ','], #1000
        "05121" : [1, None, None, ','], #1000
        "05261" : [1, None, None, ','], #18, 1000
        "06426" : [1, None, None, ','], #2000
        "06453" : [1, None, None, ','], #300
        "06995" : [1, None, None, ','], #900
        "07162" : [1, None, None, ','],
        "07859" : [1, None, None, ','],
        "07879" : [1, None, None, ','],
        "07910" : [1, None, None, ','], #10, 320
        "08215" : [1, None, None, ','], #400
        "08219" : [1, None, None, ';'], #5000
        "08378" : [1, None, None, ';'], #220
        "08405" : [1, None, None, ';'],
        "08434" : [1, None, None, ';'],
        "08455" : [1, None, None, ';'], #90
    }
    
    print("============================ *** ============================")
    print("=                 PREPROCESSING DATASET AFDB                =") 
    print("============================ *** ============================")
    for record in records :
        print("[INFO] processing recod %s..." % record)
        start = records[record][0]
        stop = records[record][1]
        separator = records[record][3]
        preprocessing_AFDB(record, start=start, stop=stop, sep=separator, fs=250)


    print("============================ *** ============================")
    print("=                PREPROCESSING DATASET NSRDB                =") 
    print("============================ *** ============================")
    nsrdb_dir = os.listdir("dataset/NSRDB")
    for record in nsrdb_dir :
        print("[INFO] processing recod %s..." % record)
        preprocessing_NSRDB(record)
    
    
    print("============================ *** ============================")
    print("=               BALANCING PER-RECORD DATASET                =") 
    print("============================ *** ============================")
    for record in records :
        n_samples = records[record][2]
        print("[INFO] balancing dataset recod %s..." % record)
        balancing_dataset(record, n_samples)
        

    print("============================ *** ============================")    
    print("=              BALANCING & MERGING ALL DATASET              =") 
    print("============================ *** ============================") 
    merging_dataset(n_samples = 4000)
    
        
    print("============================ *** ============================") 
    print("=                         DENOISING                         =") 
    print("============================ *** ============================") 
    denoising()

    
    print("============================ *** ============================") 
    print("=                      CLASSIFICATION                       =") 
    print("============================ *** ============================") 
    # isi dengan 'lms', 'nlms', 'rls' untuk memilih sumber dataset dari hasil denoising tsb.
    # isi dengan 'ori' jika ingin menggunakan original dataset
    # isi dengan 'noised' jika ingin menggunakan noised dataset
    classification(denoised = 'lms', EPOCHS = 16, BATCH_SIZE = 128)
    classification(denoised = 'nlms', EPOCHS = 16, BATCH_SIZE = 12)
    classification(denoised = 'rls', EPOCHS = 16, BATCH_SIZE = 128)
#     classification(denoised = 'ori', EPOCHS = 16, BATCH_SIZE = 128)
#     classification(denoised = 'noised', EPOCHS = 16, BATCH_SIZE = 128)