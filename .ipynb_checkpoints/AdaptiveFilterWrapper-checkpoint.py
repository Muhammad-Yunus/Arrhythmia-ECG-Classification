import numpy as np
import padasip as pa
from padasip.filters import AdaptiveFilter


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
            
            snr = self.calc_snr(y, x)
            mse = pa.misc.MSE(e)
            rmse = pa.misc.RMSE(e)
            if i % int(signals.shape[0]/10) == 0 :
                print("sample %d \t : SNR %.4f, MSE %.4f, RMSE %.4f" % (i, snr, mse, rmse))
                
    def fit_transform(self, noised_signals, signals):
        self.fit(noised_signals, signals)
        
        return self.transform(noised_signals)
    
    def transform(self, signals):
        vec = np.vectorize(self._model.predict)   
        result = vec(signals)
        
        return result
        
    def calc_snr(signal, noised_signal):
        #convert to dB 
        noise = signal - noised_signal
        std_noise = np.std(noise)
        signal_avg = np.mean(signal)

        SNR  = signal_avg/std_noise

        return SNR