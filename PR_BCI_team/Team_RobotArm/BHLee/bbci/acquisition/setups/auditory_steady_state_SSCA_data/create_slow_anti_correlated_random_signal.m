function [y, t] = create_slow_anti_correlated_random_signal(T, fs, lp_cutoff_freq, buffer_length)
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

if nargin < 4
    buffer_length = 60*fs;
end

N = T*fs;
% random white noise, buffered to avoid border effects of the filter
y = randn(2,N+2*buffer_length);
% lowpass filter the noise
WpWs = [lp_cutoff_freq, lp_cutoff_freq*4]/fs*2;
[filt_ord, Wn] = buttord(WpWs(1), WpWs(2), 3, 20);
[filt.b, filt.a] = butter(filt_ord,Wn);
y(1,:) = filtfilt(filt.b, filt.a, y(1,:));
y(1,:) = y(1,:) - mean(y(1,:));
y(2,:) = -y(1,:);
y = y(:,(buffer_length+1):(end-buffer_length+1));
t = (0:N)/fs;


% apply a fade-in / fade-out window
fading_time = 10; % interval in which the tone is faded in and out at the beginning and at the end
fading_window_end = 0.5 + 0.5*cos(linspace(0, pi, fading_time*fs));
fading_window_start = fading_window_end(end:-1:1);
fading_window = [fading_window_start ones(1,length(y)-2*length(fading_window_start)) fading_window_end];
y = y - repmat(mean(y,2), 1, length(y));
if size(y,1) == 2
    fading_window = [fading_window; fading_window];
end
y = y.*fading_window;


% histogramm equalization
for k=1:2
    [foo, sort_indices] = sort(y(k,:));
    [foo, y(k,:)] = sort(sort_indices);
end
