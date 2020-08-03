function noise = stim_filteredNoise(sample_freq, duration, filter_type, lowerBound, upperBound, rampon, rampoff)
%stim_filteredNoise
%
%Synopsis:
% [noise]= stim_oddballAuditory(sample_freq, duration, filter_type, lowerBound, upperBound, rampon, rampoff)
%
%Arguments:
% sample_freq: preset samplefrequency used
% duration: duration of the noise
% filter_type: type of filter used
%           1 (lowpass), (2) highpass, (3) bandpass, (4) notch
% lowerBound: lowest frequency
% upperBound: highest frequency
% rampon: time (msec) for ramp onset
% rampoff: time (msec) for ramp offset

% set general variables
sf = sample_freq;  % sample frequency
nf = sf / 2; % nyquist frequency
d = duration;     % duration
n = sf * d;  % number of samples
nh = n / 2;  % half number of samples

% =========================================================================
% set variables for filter
lf = lowerBound;   % lowest frequency
hf = upperBound;   % highest frequency
lp = lf * d; % ls point in frequency domain    
hp = hf * d; % hf point in frequency domain

% design filter
switch filter_type
    case 1
    a = ['LOWPASS'];
    filter = zeros(1, n);           % initializaiton by 0
    filter(1, 1 : lp) = 1;          % filter design in real number
    filter(1, n - lp : n) = 1;      % filter design in imaginary number
    case 2        
    a = ['HIGHPASS'];
    filter = ones(1, n);            % initializaiton by 1
    filter(1, 1 : hp) = 0;          % filter design in real number
    filter(1, n - hp : n) = 0;      % filter design in imaginary number
    case 3
    a = ['BANDPASS'];
    filter = zeros(1, n);           % initializaiton by 0
    filter(1, lp : hp) = 1;         % filter design in real number
    filter(1, n - hp : n - lp) = 1; % filter design in imaginary number
    case 4
    a = ['NOTCH'];
    filter = ones(1, n);
    filter(1, lp : hp) = 0;
    filter(1, n - hp : n - lp) = 0;
end

% =========================================================================
% make noise
rand('state',sum(100 * clock));  % initialize random seed
noise = randn(1, n);             % Gausian noise
noise = noise / max(abs(noise)); % -1 to 1 normalization

% do filter
s = fft(noise);                  % FFT
s = s .* filter;                 % filtering
s = ifft(s);                     % inverse FFT
s = real(s);

Non= round(rampon/1000*sf);
ramp= sin((1:Non)*pi/Non/2).^2;
s(1:Non)= s(1:Non) .* ramp;
Noff= round(rampoff/1000*sf);
ramp= cos((1:Noff)*pi/Noff/2).^2;
s(end-Noff+1:end)= s(end-Noff+1:end) .* ramp;

noise = s;

x = linspace(0, nf, nh);
t = fft(s);
t = t .* conj(t);
semilogy(x, t(1,1:nh) ./ max(t));  xlabel('frequency (Hz)');  title('spectrum: filtered noise'); 

end