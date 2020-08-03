function epo_tf= proc_tf_stft(epo, band, N, step)
% epo_tf= proc_tf_stft(epo, <N=epo.fs, step=N/2>)
%
% IN  epo  - data structure of epoched data
%     band  - frequency band of interest [lowerHz, upperHz]
%     N    - length of window [sample points]
%     step - window step size, default N/2
%
% OUT epo_tf - time-frequency representation [4-d data field]

if ~exist('N','var') | isempty(N),
  N= epo.fs;
end
if ~exist('step','var'), step= N/2; end

if ndims(epo.x)>3, 
  error('data must be 2- or 3-dimensional');
end

[bInd, bFreq]= getBandIndices(band, epo.fs, N);
[T, nChans, nEpochs]= size(epo.x);
nWindows= 1 + max(0, floor((T-N)/step));
epo_tf= copyStruct(epo, 'x');
epo_tf.x= zeros([length(bInd), nWindows, nChans, nEpochs]);
epo_tf.f= bFreq;
epo_tf.t= N/2:step:T;
for ee= 1:nEpochs,
  for cc= 1:nChans,
    qq= STFT(epo.x(:,cc,ee), step, N);
    epo_tf.x(:,:,cc,ee)= 20*log10(abs(qq(bInd,:)));
  end
end



function TF_matrix = STFT(series, step, window_size);
% ===================================================
% Short-Time Fourier Transform (STFT)
% TF_matrix = STFT(series, step, window_size)
%
% IN    series  -   time series
%       step    -   step size
%                   if step size equal to window size -> no overlap 
%       window_size -   window size
%                       a wide window   -> good frequency resolution but poor time resolution
%                       a narrow window -> good time resolution but poor frequency resolution.
%
% OUT   TF_matrix - time-frequency analysis result
%                   column ->frequency
%                   row->time
% ===================================================
if ~exist('series', 'var')|isempty(series)|~exist('step','var')|isempty(step)|~exist('window_size','var')|isempty(window_size)
    error('Not enough input arguments');
end

s_point = (1:step:length(series)-window_size);
e_point = (window_size:step:length(series));
tt = zeros(window_size, length(s_point));
for i = 1:1:length(s_point)
    tt(:,i) = series(s_point(i):e_point(i));    
end
TF_matrix = fft(tt);

