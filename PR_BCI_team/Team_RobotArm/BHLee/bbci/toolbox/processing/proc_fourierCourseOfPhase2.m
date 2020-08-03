function out= proc_fourierCourseOfPhase2(dat, freq, win, step)
%epo= proc_fourierCourseOfPhase(epo, freq, <N/win, step=N>)
%
% calculate the time course of the phase of fourier coefficients
% at a specified frequency
% 
% IN   epo   - data structure of epoched data
%      freq  - frequency of interest [Hz]
%      win   - fourier window
%      N     - length of window -> square (boxcar) window is used
%      step  - window step size, default: N (i.e., non-overlapping windows)
%
% OUT  epo   - updated data structure

% bb, ida.first.fhg.de


[T, nChans, nEvents]= size(dat.x);
if ~exist('win','var'), win=T; end
if length(win)==1,
  win= ones(win,1);
end
N= length(win);
if N>T, error('window longer than signals'); end

[mm,bInd]= min(abs(freq-[(0:N/2)*dat.fs/N]));

nWindows= 1 + max(0, floor((T-N)/step));
iv= 1:N;
out= copy_struct(dat, 'not', 'x','t');
out.x= zeros([nWindows, nChans*nEvents]);
ExpWin= win(:)' .* exp(-2*pi*i*(bInd-1)*(0:N-1)/N);
ExpWinM1= win(:)' .* exp(-2*pi*i*(bInd-2)*(0:N-1)/N);
ExpWinP1= win(:)' .* exp(-2*pi*i*(bInd)*(0:N-1)/N);
for wi= 1:nWindows,
  for ce= 1:nChans*nEvents,
    X(1)= ExpWinM1 * dat.x(iv, ce);
    X(2)= ExpWin * dat.x(iv, ce);
    X(3)= ExpWinP1 * dat.x(iv, ce);
%    [mm,mi]= max(abs(X));
%    out.x(wi,ce)= angle(X(mi));
    out.x(wi,ce)= angle(mean(X));
  end
  out.t(wi)= dat.t(iv(end));
  iv= iv + step;
end
out.x= reshape(out.x, [nWindows, nChans, nEvents]);
out.yUnit= 'rad';
