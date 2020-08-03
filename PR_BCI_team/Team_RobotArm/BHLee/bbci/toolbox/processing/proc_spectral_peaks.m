function [peak_freqs, cnt_fft_normed, cnt_fft, P] = proc_spectral_peaks(cnt, varargin)
% Compute the peak frequencies in the given band, i.e. local maxima in the
% power spectrum (after normalization with best-fit exponential).

%% params
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt ...
    ,'channel', 'Pz' ...
    ,'normalization_range', [1,40] ... % range in which the normalized spectrum is computed 
    ,'peak_range', [8,13] ... % range in which peaks are determined
    ,'frequency_resolution', 0.5 ... % in Hz
);

%% reduce the cnt
cnt = proc_selectChannels(cnt, opt.channel);

%% compute the frequency spectrum 
window_length_ms = 1000/opt.frequency_resolution;
window_length = window_length_ms*cnt.fs/1000;
window_length = min(window_length, size(cnt.x,1));
cnt_fft = proc_spectrum(cnt, opt.normalization_range, 'win', kaiser(window_length));
freqs = cnt_fft.t;

% % sanity check
% figure, plot(freqs, cnt_fft.x)


%% fit exponential
% exponential is fitted by fitting a linear function to the log-log representation of
% the frequency spectrum
P = polyfit(log(freqs), log(cnt_fft.x)', 1);
% P(2) = P(2) + (log(cnt_fft.x(1))- (log(freqs(1))*P(1) + P(2)));

% % sanity check
% figure 
% plot(log(freqs), log(cnt_fft.x), 'b')
% hold on
% plot(log(freqs), P(1)*log(cnt_fft.t) + P(2), 'r')

%% normalize the spectrum

cnt_fft_normed = cnt_fft;
cnt_fft_normed.x = cnt_fft.x' ./ exp( P(1)*log(freqs) + P(2));

% figure, plot(freqs, cnt_fft_normed.x)

%% find the peak
peak_window_idx = find((freqs >= opt.peak_range(1)) & (freqs <= opt.peak_range(2)));
peak_window_freqs = freqs(peak_window_idx);
x = cnt_fft_normed.x(peak_window_idx);
local_peaks_idx = local_max(x);
peak_freqs = peak_window_freqs(local_peaks_idx);
