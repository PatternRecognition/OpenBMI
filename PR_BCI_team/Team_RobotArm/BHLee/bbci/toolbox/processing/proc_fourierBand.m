function dat= proc_fourierBand(dat, band, N)
%epo= proc_fourierBand(epo, band, N)
%
% calculate complex fourier coefficients in a specified band
% only one fourier transform is calculated per signal, i.e. no 
% overlap-and-add method for signals longer than the ft window is applied.
% 
% IN   epo   - data structure of epoched data
%      band  - frequency band of interest [lowerHz, upperHz]
%      N     - length of square (boxcar) window
%
% OUT  epo   - updated data structure
%
% SEE proc_fourierBandMagnitude, proc_fourierBandEnergy

% bb, ida.first.fhg.de


[T, nChans, nEvents]= size(dat.x);
if ~exist('N','var'), N= 2^nextpow2(dat.fs); end

[bInd, bFreq]= getBandIndices(band, dat.fs, N);

x= reshape(dat.x, [T, nChans*nEvents]);
X= fft(x(max(1,T-N+1):T, :), N);
X= X(bInd,:);
dat.x= reshape(X, [size(X,1) nChans nEvents]);
dat.t= bFreq;
