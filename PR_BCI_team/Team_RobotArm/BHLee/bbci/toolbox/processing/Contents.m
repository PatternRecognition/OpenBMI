%processing - processing and feature extraction methods
%
%proc_* functions transform feature vector structs to feature vector
%structs. Some functions expect a more specific structure of the
%feature vectors, e.g., continuous time series, epoched time series,...
%
%   proc_appendChannels        - append channels from one to another data set
%   proc_appendEpochs          - append epochs to an epoch structure
%   proc_arCoefs               - calculate AR coefficients
%   proc_arCoefsPlusVar        - as above but add signal variance to feature
%   proc_average               - calculate averages for each class
%   proc_bandPower             - calculate power in a given frequency band
%   proc_baseline              - subtract epochwise the mean of a reference
%                                time interval
%   proc_csp                   - projection on common spatial filters
%   proc_filt                  - apply digital filter (matlab filter function)
%   proc_filtfilt              - zero-phase filter (matlab filtfilt function)
%   proc_filtByFFT             - overlap-and-add FFT filtering technique
%   proc_filtNotchbyFFT        - notch filter (eg. to remove 50 resp. 60 Hz)
%   proc_flaten                - reshape feature matrix to feature vector
%   proc_fourierBand           - complex fourier coefficients
%   proc_fourierBandEnergy     - energy within a given spectral band
%   proc_fourierBandMagnitude  - absolute value of fourier coefficients
%   proc_fourierBandReal       - real + imag part of fourier coefficients
%   proc_jumpingMeans          - means in non-overlap. windows (= downsampling)
%   proc_linearDerivation      - calculate linear combinations of channels
%   proc_movingAverage         - calculate moving average of input signals
%   proc_r_square              - statistical r^2 measure
%   proc_rcCoefsPlusVar        - reflections coefficients (similar to AR coef)
%   proc_rectifyChannels       - convert all samples to absolute values
%   proc_selectChannels        - select specific channels and discard the rest
%   proc_selectIval            - select specific time interval
%   proc_spectrum              - calculate power spectral density
%   proc_squareChannels        - square all signals samplewise
%   proc_subtractMovingAverage - subtract MA (high-pass filtering)
%   proc_t_scale               - calculated t-scaled class difference
%   proc_variance              - variance in each channel
%   proc_pr_isomap             - isomap
%   proc_pr_lle                - lle
%   proc_pr_trPCA              - calculate task related PCA
