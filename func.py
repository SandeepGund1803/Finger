import numpy as np
from scipy.signal import resample
from scipy.signal import find_peaks
import cv2
from moviepy.editor import *
import moviepy
#from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Activation, add
#from tensorflow.keras.models import Model
import os

class Finger():

    def __init__(self,vid_path):
        self.vid_path = vid_path

    def Raw_PPG(self):   #Eextract the raw ppg signal from video of 30 sec.
        
        vidcap = cv2.VideoCapture(self.vid_path)
        clip = moviepy.editor.VideoFileClip(self.vid_path)
        seconds = clip.duration
        
        rate1=1/125
        rate2=1/30
        frame1=0
        frame2=0
        red_dc=[]
        blue_dc=[]
        red_ac=[]
        blue_ac=[]
        red_dc1=[]
        while True:
            if frame1 < 8.191:
                vidcap.set(cv2.CAP_PROP_POS_MSEC,frame1*1000)   #Extracting frames fron video   
                success,image = vidcap.read()
                if success:
                    width = 1024
                    height = 1
                    dsize = (width, height)
                    image = cv2.resize(image,dsize)
                    avg_red = np.mean(image[:,:,0])
                    red_dc1.append(avg_red)  
                frame1+=rate1
                
            else:                                                       #Used to extract Blood pressure
                vidcap.set(cv2.CAP_PROP_POS_MSEC,frame2*1000)   #Extracting frames fron video   
                success,image = vidcap.read()
                if success:
                    width = 640
                    height = 320
                    dsize = (width, height)                    #Fix the size of frame
                    image = cv2.resize(image,dsize)
                    Mr = np.mean(image[:,:,0])    # mean of red colour component   Red DC Values
                    Mb = np.mean(image[:,:,2])    # mean of blue colour component  blue DC values
                    Sdr = np.std(image[:,:,0])    # standerd deviation of red colour comp.    Red AC value
                    Sdb = np.mean(image[:,:,2])    # standerd deviation of blue colour comp.   Blue AC value
                
                    red_dc.append(Mr)
                    blue_dc.append(Mb)
                    red_ac.append(Sdr)
                    blue_ac.append(Sdb)
                frame2+=rate2
                if frame2>seconds:
                    break

        return red_dc1,red_dc,blue_dc,red_ac,blue_ac


    def Filtered_PPG(self,raw_ppg_values,freqMin, freqMax,fps):     #converting raw ppg to filtered ppg that is removing noise from signal
        total_frames = len(raw_ppg_values)
        duration = total_frames / fps
        sig = raw_ppg_values
        sig = resample(sig, (len(sig) * 2) + int(0.1 * len(sig)))
        n = len(sig)
        dt = duration / n
        t = np.linspace(0, duration, n)

        # --- GET FREQUENCIES VECTOR
        fcomp = np.fft.fft(sig,n)
        freq = (1/(dt*n)) * np.arange(n)
        # --- SELECT SPECIFIED FREQUENCY FOR FILTERING
        ffilt = fcomp
        freqMin = freqMin #Hz
        freqMax = freqMax    #Hz
        for f in range(1,int(np.floor(len(freq)/2))):
            tempFreq = freq[f]
            if tempFreq <= freqMin or tempFreq >= freqMax:
                ffilt[f] = 0
                ffilt[-f] = 0
            
  	    # --- IFFT TO GET FILTERED SIGNAL
        sigfilt = np.real(np.fft.ifft(ffilt))
        for nc in range(len(sigfilt)):
            sigfilt[nc] = sigfilt[nc].real
        
  	    # --- PAD FILTERED SIGNAL
        nInds = int(len(sigfilt)*0.02)      # percetage of padding
        sigfilt[0:nInds] = sigfilt[nInds*2:nInds:-1]
        sigfilt[-1-nInds:-1] = sigfilt[-1-(nInds*2):-1-nInds]
        
        return sigfilt,duration,t

    def Find_HR_RR(self,red_dc,freqMin, freqMax,fps):     #To estimate Heart rate and Respiration rate
        PPG_Signal,duration,t = self.Filtered_PPG(red_dc,freqMin, freqMax,fps)

        # MAX PEAKS
        horzdist = 12
        peaks = find_peaks(PPG_Signal, height=1, distance=horzdist)
        height = peaks[1]['peak_heights']
        peak_pos = t[peaks[0]]
  
        # MIN PEAKS
        invsig = PPG_Signal * -1
        minima = find_peaks(invsig.real, distance=horzdist)
        min_pos = t[minima[0]]
        min_height = invsig[minima[0]]
        min_height = min_height * -1
  
        # --- BPM ---
        MAX = len(peak_pos)
        MIN = len(min_pos)
        Result = (MIN + MAX) / 2
        Result = int(np.around(60 * Result / duration))

        return Result

    def spo2_para(self,sigfilt,t):          #macking useful parameter for spo2 from filtered ppg 
        # MAX PEAKS
        horzdist = 12
        peaks = find_peaks(sigfilt, height=1, distance=horzdist)
        height = peaks[1]['peak_heights']
        peak_pos = t[peaks[0]]
  
        # MIN PEAKS
        invsig = sigfilt * -1
        minima = find_peaks(invsig.real, distance=horzdist)
        min_pos = t[minima[0]]
        min_height = invsig[minima[0]]
        min_height = min_height * -1
  
        # --- BPM ---
        MAX = len(peak_pos)
        MIN = len(min_pos)
        result = (MIN + MAX) / 2
   
        return result

    def Find_SpO2(self,red_dc,blue_dc,red_ac,blue_ac):     #to estimate spo2

        PPG,duration,t = self.Filtered_PPG(red_dc,freqMin = 0.7, freqMax = 4,fps=30)
        DC_red = self.spo2_para(PPG,t)
        PPG,duration,t = self.Filtered_PPG(blue_dc,freqMin = 0.7, freqMax = 4,fps=30)
        DC_blue = self.spo2_para(PPG,t)
        PPG,duration,t = self.Filtered_PPG(red_ac,freqMin = 0.7, freqMax = 4,fps=30)
        AC_red = self.spo2_para(PPG,t)
        PPG,duration,t = self.Filtered_PPG(blue_ac,freqMin = 0.7, freqMax = 4,fps=30)
        AC_blue = self.spo2_para(PPG,t)

        spo2 = int(np.ceil(100 - (5 * ((AC_red/DC_red)/(AC_blue/DC_blue)))))

        return spo2

    def Find_HRV(self,red_dc,freqMin, freqMax,fps):   # To estimate Heart rate variation
        PPG_Signal,duration,t = self.Filtered_PPG(red_dc,freqMin, freqMax,fps)

        # MAX PEAKS
        horzdist = 12
        peaks = find_peaks(PPG_Signal, height=1, distance=horzdist)
        height = peaks[1]['peak_heights']
        peak_pos = t[peaks[0]]
   
        peak_times = peak_pos * 1000
        intervals = np.diff(peak_times)
        hrv = np.diff(intervals)
        HRV = int(np.floor(np.std(hrv)))

        return HRV

    def Find_Stree(self,HRV):              # To estimate Stress index from HRV
        if HRV < 50:
            SI = " High"
        elif 50 < HRV < 100:
            SI = "  Normal"
        else:
            SI = "  Low"
        return  SI

"""

    def UNetDS64(self,length, n_channel=1):
        '''
            Deeply supervised U-Net with kernels multiples of 64
    
        Arguments:
            length {int} -- length of the input signal
    
        Keyword Arguments:
            n_channel {int} -- number of channels in the output (default: {1})
    
        Returns:
            keras.model -- created model
        '''
    
        x = 64

        inputs = Input((length, n_channel))
        conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)

        conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling1D(pool_size=2)(conv2)

        conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling1D(pool_size=2)(conv3)

        conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling1D(pool_size=2)(conv4)

        conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
    
        level4 = Conv1D(1, 1, name="level4")(conv5)

        up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
        conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)
    
        level3 = Conv1D(1, 1, name="level3")(conv6)

        up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
        conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)
    
        level2 = Conv1D(1, 1, name="level2")(conv7)

        up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
        conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)
    
        level1 = Conv1D(1, 1, name="level1")(conv8)

        up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
        conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        out = Conv1D(1, 1, name="out")(conv9)

        model = Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])

        return model

    def MultiResUNet1D(self,length, n_channel=1):
        '''
           1D MultiResUNet
    
        Arguments:
            length {int} -- length of the input signal
    
        Keyword Arguments:
            n_channel {int} -- number of channels in the output (default: {1})
    
        Returns:
            keras.model -- created model
        '''
        def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
        
            kernel = 3

            x = Conv1D(filters, kernel,  padding=padding)(x)
            x = BatchNormalization()(x)

            if(activation == None):
                return x

            x = Activation(activation, name=name)(x)
            return x
        
        def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
 
            x = UpSampling1D(size=2)(x)        
            x = BatchNormalization()(x)
        
            return x
        
        def MultiResBlock(U, inp, alpha = 2.5):
            '''
            MultiRes Block
        
            Arguments:
                U {int} -- Number of filters in a corrsponding UNet stage
                inp {keras layer} -- input layer 
        
            Returns:
                [keras layer] -- [output layer]
            '''
            W = alpha * U

            shortcut = inp

            shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
                                int(W*0.5), 1, 1, activation=None, padding='same')

            conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
                                activation='relu', padding='same')

            conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
                                activation='relu', padding='same')

            conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
                                activation='relu', padding='same')

            out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
            out = BatchNormalization()(out)

            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)

            return out

        def ResPath(filters, length, inp):
            '''
            ResPath
        
            Arguments:
                filters {int} -- [description]
                length {int} -- length of ResPath
                inp {keras layer} -- input layer 
        
            Returns:
                [keras layer] -- [output layer]
            '''
            
            shortcut = inp
            shortcut = conv2d_bn(shortcut, filters, 1, 1,
                                activation=None, padding='same')

            out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

            out = add([shortcut, out])
            out = Activation('relu')(out)
            out = BatchNormalization()(out)

            for i in range(length-1):

                shortcut = out
                shortcut = conv2d_bn(shortcut, filters, 1, 1,
                                    activation=None, padding='same')

                out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

                out = add([shortcut, out])
                out = Activation('relu')(out)
                out = BatchNormalization()(out)

            return out

        inputs = Input((length, n_channel))

        mresblock1 = MultiResBlock(32, inputs)
        pool1 = MaxPooling1D(pool_size=2)(mresblock1)
        mresblock1 = ResPath(32, 4, mresblock1)

        mresblock2 = MultiResBlock(32*2, pool1)
        pool2 = MaxPooling1D(pool_size=2)(mresblock2)
        mresblock2 = ResPath(32*2, 3, mresblock2)

        mresblock3 = MultiResBlock(32*4, pool2)
        pool3 = MaxPooling1D(pool_size=2)(mresblock3)
        mresblock3 = ResPath(32*4, 2, mresblock3)

        mresblock4 = MultiResBlock(32*8, pool3)
        pool4 = MaxPooling1D(pool_size=2)(mresblock4)
        mresblock4 = ResPath(32*8, 1, mresblock4)

        mresblock5 = MultiResBlock(32*16, pool4)

        up6 = concatenate([UpSampling1D(size=2)(mresblock5), mresblock4], axis=-1)
        mresblock6 = MultiResBlock(32*8, up6)

        up7 = concatenate([UpSampling1D(size=2)(mresblock6), mresblock3], axis=-1)
        mresblock7 = MultiResBlock(32*4, up7)

        up8 = concatenate([UpSampling1D(size=2)(mresblock7), mresblock2], axis=-1)
        mresblock8 = MultiResBlock(32*2, up8)

        up9 = concatenate([UpSampling1D(size=2)(mresblock8), mresblock1], axis=-1)
        mresblock9 = MultiResBlock(32, up9)

        conv10 = Conv1D(1, 1)(mresblock9)
    
        model = Model(inputs=[inputs], outputs=[conv10])

        return model

    def Find_BP(self,red_dc1,freqMin, freqMax,fps):     #To estimate Blood pressure
        raw_sig = red_dc1
        sigfilt,duration,t = self.Filtered_PPG(raw_sig,freqMin, freqMax,fps)

        normalizedData = (sigfilt-np.min(sigfilt))/(np.max(sigfilt)-np.min(sigfilt))     #Normalise the filtered ppg signal

        normalized_ppg = normalizedData[:1024]                             #used ony 1st 1024 ppg signals
        normalized_ppg = np.array(normalized_ppg,float)                    #creating array of normalised ppg
        normalized_ppg = normalized_ppg.reshape((-1,1024,1))

        max_abp = 199.9479008990615                                #used to denormalise predicted abp signal
        min_abp = 50.0                                             
        length = 1024

        mdl1 = self.UNetDS64(length)                                             # creating approximation network
        mdl1.load_weights(os.path.join('models','ApproximateNetwork.h5'))   # loading weights
        Y_test_pred_approximate = mdl1.predict(normalized_ppg,verbose=1)            # predicting approximate abp waveform
        mdl2 = self.MultiResUNet1D(length)                                       # creating refinement network
        mdl2.load_weights(os.path.join('models','RefinementNetwork.h5'))    # loading weights
        Y_test_pred = mdl2.predict(Y_test_pred_approximate[0],verbose=1)    # predicting abp waveform
        abp_signal_pred = Y_test_pred[0] * max_abp + min_abp
        SBP = int(np.ceil(max(abp_signal_pred)-20))
        DBP = int(np.ceil(min(abp_signal_pred)+25))

        return SBP,DBP
"""