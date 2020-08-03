function out= proc_fourierCourseOfBandMagnitude(dat, band, win, step)
%epo= proc_fourierCourseOfBandMagnitude(epo, band, <N/win, step=N>)
%
% calculate the time course of the magnitude of fourier coefficients
% in a specified frequency band
% 
% IN   epo   - data structure of epoched data
%      band  - frequency band of interest [lowerHz, upperHz]
%      win   - fourier window
%      N     - length of window -> square (boxcar) window is used
%      step  - window step size, default: N (i.e., non-overlapping windows)
%
% OUT  epo   - updated data structure
%
% SEE proc_fourierBandMagnitude

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

nWindows= 1 + max(0, floor((T-N)/step));
iv= 1:N;
out= copy_struct(dat, 'not', 'x','t');
out.x= zeros([nWindows, length(bInd), nChans, nEvents]);
for wi= 1:nWindows,
  X= fft(dat.x(iv, :).*Win);
  %xo= abs( 10*log10( abs(X(bInd,:))+eps ));
  xo= abs(X(bInd,:));
  out.x(wi,:,:,:)= reshape(xo, [1, length(bInd), nChans, nEvents]);
  out.t(wi)= dat.t(iv(end));
  iv= iv + step;
end
out.x= reshape(out.x, [nWindows*length(bInd), nChans, nEvents]);
out.yUnit= 'power';
