function dat= proc_filtBruteFFT(dat, band, win, len)
%dat= procFiltBruteFFT(dat, band, win/N, len)
%
% IN   dat   - data structure of continuous or epoched data
%      band - [lower upper] in Hz
%      win  - window applied to padded signal before FFT
%      N    - length of FFT [must be >= size(y,1)] -> one-sided
%             cosine window is used which may be inappropriate
%      len  - length of the tailing interval to extract [in msec]
%
% OUT  dat   - updated data structure

% bb, ida.first.fhg.de


[T, nChans, nEvents]= size(dat.x);
if nargin<3, win= 2^nextpow2(max(T, dat.fs)); end
if length(win)==1,
  N= win;
  win= 1-cos((1:N)/N*pi);
else
  N= length(win);
end
if nargin<4, 
  ival= 1:N;
else
  lenSa= len*dat.fs/1000;
  ival= N-lenSa+1:N;
  if isfield(dat,'t')
    dat.t = dat.t(round((1:lenSa)+length(dat.t)-lenSa));
  end
end

bInd= getBandIndices(band, dat.fs, N);
specMask= zeros(N, nChans*nEvents);
if bInd(1)==1,
  bIndNeg= N+2-bInd(2:end);
else
  bIndNeg= N+2-bInd;
end
specMask([bInd bIndNeg],:)= 1;

iv= max(1,T-N+1):T;
Win= repmat(win(:), 1, nChans*nEvents);
pad= zeros(N-length(iv), nChans*nEvents);

X= fft([pad; dat.x(iv,:)].*Win, N);
xf= real(ifft( X.*specMask ));
dat.x= reshape(xf(ival,:), [length(ival) nChans nEvents]);
