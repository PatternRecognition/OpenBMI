function dat= proc_fourierBandReal(dat, band, win)
%epo= procFourierBandReal(epo, band, <N/win>)
%
% calculate fourier coefficients in a specified band and concatenate
% real and imaginary parts in a real valued vector
% only one fourier transform is calculated per signal, i.e. no 
% overlap-and-add method for signals longer than the ft window is applied.
% 
% IN   epo   - data structure of epoched data
%      band  - frequency band of interest [lowerHz, upperHz]
%      win   - fourier window
%      N     - length of window -> one sided cosine window is used,
%              with is inappropriate in many cases
%
% OUT  epo   - updated data structure
%
% SEE proc_fourierBand, proc_fourierBandMagnitude, proc_fourierBandEnergy

% bb, ida.first.fhg.de


[T, nChans, nEvents]= size(dat.x);
if ~exist('win','var'), 
  win= ones(1, 2^nextpow2(dat.fs));
elseif length(win)==1,
  win= 1-cos((1:win)/win*pi);
end
winLen= length(win);
N= 2^nextpow2(winLen);
iv= max(1,T-winLen+1):T;

if length(band)==2,
  bInd= getBandIndices(band, dat.fs, N);
else
  bInd= band;
end
bPos= bInd(find(bInd>1));
Win= repmat(win(:), 1, nChans*nEvents);
pad= zeros(winLen-length(iv), nChans*nEvents);

%Y= fft([pad; dat.x(iv, :)].*Win, N) / norm(win);
Y= fft([pad; dat.x(iv, :)].*Win, N);
xo= [real(Y(bInd,:)); imag(Y(bPos,:))];
dat.x= reshape(xo, [size(xo,1) nChans nEvents]);
