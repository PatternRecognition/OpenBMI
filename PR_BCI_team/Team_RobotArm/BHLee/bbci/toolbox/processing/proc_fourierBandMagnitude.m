function dat= proc_fourierBandMagnitude(dat, band, win)
%epo= proc_fourierBandMagnitude(epo, band, <N/win>)
%
% calculate the magnitude of fourier coefficients in a specified band
% only one fourier transform is calculated per signal, i.e. no 
% overlap-and-add method for signals longer than the ft window is applied.
% 
% IN   epo   - data structure of epoched data
%      band  - frequency band of interest [lowerHz, upperHz]
%      win   - fourier window
%      N     - length of window -> square (boxcar) window is used
%
% OUT  epo   - updated data structure
%
% SEE proc_fourierBand, proc_fourierBandReal, proc_fourierBandEnergy,
%     proc_fourierCourseOfBandMagnitude

% bb, ida.first.fhg.de


[T, nChans, nEvents]= size(dat.x);
if ~exist('win','var'), win=T; end
if length(win)==1,
  win= ones(win,1);
end
N= length(win);
if N>T, error('window longer than signals'); end

Win= repmat(win(:), 1, nChans*nEvents);

[bInd, bFreq]= getBandIndices(band, dat.fs, N);
X= fft(dat.x(T-N+1:T, :).*Win);
xo= abs(X(bInd,:)); %% / norm(win);
dat.x= reshape(xo, [size(xo,1) nChans nEvents]);
dat.t= bFreq;
dat.xUnit= 'Hz';
dat.yUnit= 'power';



return 

%% with padding:

[T, nChans, nEvents]= size(dat.x);
if ~exist('win','var'), win=T; end
if length(win)==1,
  win= ones(win,1);
end
N= length(win);
iv= max(1,T-N+1):T;

Win= repmat(win(:), 1, nChans*nEvents);
pad= zeros(N-length(iv), nChans*nEvents);

[bInd bFreq]= getBandIndices(band, dat.fs, N);
X= fft([pad; dat.x(iv, :)].*Win, N);
xo= abs(X(bInd,:)); %% / norm(win);
dat.x= reshape(xo, [size(xo,1) nChans nEvents]);
dat.t= bFreq;
