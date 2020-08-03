function [y, t] = create_slow_random_signal(T, fs, lp_cutoff_freq)
% [y, t] = create_slow_random_signal(T, fs, lp_cutoff_freq)
%
% In:
% T - length of resulting signal, given in seconds
% fs - sampling rate
% lp_cutoff_freq - cutoff frequency of the low-pass filter
%
% Out:
% y - the low-pass filtered noise
% t - vector of time indices, same length as y
%
% 12-2011 sven.daehne@tu-berlin.de

N = T*fs;
buffer_length = 60*fs;
% random white noise, buffered to avoid border effects of the filter
y = randn(1,N+2*buffer_length);
% lowpass filter the noise
WpWs = [lp_cutoff_freq, lp_cutoff_freq*4]/fs*2;
[filt_ord, Wn] = buttord(WpWs(1), WpWs(2), 3, 20);
[filt.b, filt.a] = butter(filt_ord,Wn);
y = filtfilt(filt.b, filt.a, y);
y = y(1,(buffer_length+1):(end-buffer_length+1));
t = (0:N)/fs;
    