function out= proc_fourierCourseOfBandEnergy(dat, band, win, step)
%epo= proc_fourierCourseOfBandEnergy(epo, band, <N/win, step=N>)
%
% calculate the time course of spectral energy in a specified band
% by fourier technique.
% 
% IN   epo   - data structure of epoched data
%      band  - frequency band of interest [lowerHz, upperHz]
%      win   - fourier window
%      N     - length of window -> square (boxcar) window is used
%      step  - window step size, default: N (i.e., non-overlapping windows)
%
% OUT  epo   - updated data structure
%
% SEE proc_fourierBandEnergy

% bb, ida.first.fhg.de

[T, nChans, nEvents]= size(dat.x);
if ~exist('win','var'), win=T; end
if length(win)==1,
  win= ones(win,1);
end
N= length(win);
if N>T, error('window longer than signals'); end
if ~exist('step','var'), step=N; end

if size(band,1)>1,
  out= proc_fourierCourseOfBandEnergy(dat, band(1,:), win, step);
  for ib= 2:size(band,1),
    oo= proc_fourierCourseOfBandEnergy(dat, band(ib,:), win, step);
    out.x= cat(1, out.x, oo.x);
  end
  return
end

Win= repmat(win(:), 1, nChans*nEvents);
[bInd, bFreq]= getBandIndices(band, dat.fs, N);

nWindows= 1 + max(0, floor((T-N)/step));
iv= 1:N;
out= copy_struct(dat, 'not', 'x','t');
out.x= zeros([nWindows, nChans, nEvents]);
out.t= zeros(1, nWindows);
for wi= 1:nWindows,
  X= fft(dat.x(iv, :).*Win);
  %xo= sum( 20*log10( abs(X(bInd,:))+eps ));
  xo= sum(abs(X(bInd,:)).^2, 1);
  out.x(wi,:,:)= reshape(xo, [1 nChans nEvents]);
  out.t(wi)= dat.t(iv(end));
  iv= iv + step;
end
out.yUnit= 'power';
