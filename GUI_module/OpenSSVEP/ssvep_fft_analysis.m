function output = ssvep_fft_analysis(SMT, varargin)

in = varargin{:};
in = opt_cellToStruct(in);
dat = SMT;

nCh = size(dat, 2);

freq = in.freq;
amp = zeros(nCh, length(freq));

for c = 1:nCh
    [YfreqDomain,frequencyRange] = positiveFFT(dat(:,c),in.fs); % °íÃÄ
    amp(c,:) = interp1(frequencyRange, YfreqDomain, freq);
end

if nCh ~= 1
    amp = mean(amp(:,:));
end
output = abs(amp);


function [X,freq] = positiveFFT(x,Fs)
N=length(x); %get the number of points
k=0:N-1;     %create a vector from 0 to N-1
T=N/Fs;      %get the frequency interval
freq=k/T;    %create the frequency range
X=fft(x)/N*2; % normalize the data

%only want the first half of the FFT, since it is redundant
cutOff = ceil(N/2);

%take only the first half of the spectrum
X = X(1:cutOff);
freq = freq(1:cutOff);