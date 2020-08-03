function dat= proc_fourierBandEnergy(dat, band, varargin)
%epo= proc_fourierBandEnergy(epo, band, <N/win>)
%epo= proc_fourierBandEnergy(epo, band, <OPT>)
%
% calculate the spectral energy in a specified band by fourier technique.
% only one fourier transform is calculated per signal, i.e. no 
% overlap-and-add method for signals longer than the ft window is applied.
% 
% IN   epo   - data structure of epoched data
%      band  - frequency band of interest [lowerHz, upperHz]
%      win   - fourier window
%      N     - length of window -> square (boxcar) window is used
%      OPT - struct or property/value list of optional properties
%       .win  - window for FFT, default ones(epo.fs, 1)
%              if it is a scalar, a square (boxcar) window of that size is used
%       .step - step size for shifting the window, default N/2
%
% OUT  epo   - updated data structure
%
% SEE proc_fourierBand, proc_fourierBandReal, proc_fourierBandMagnitude

% bb, ida.first.fhg.de

if ~isempty(varargin) & isnumeric(varargin{1}),
  %% arguments given as <win/N, step>
  opt.win= varargin{1};
else
  %% arguments given as <opt>
  opt= propertylist2struct(varargin{:});
end
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'win', dat.fs);

[T, nChans, nEvents]= size(dat.x);
if length(opt.win)==1,
  if opt.win>T,
    if ~isdefault.win,
      warning(['Requested window length longer than signal: ' ...
               'shortning window, no zero-padding!']);
    end
    opt.win= T;
  end
  opt.win=ones(opt.win,1);
end
N= length(opt.win);
normWin  = norm(opt.win) ;
if ~isfield(opt, 'step'), opt.step= floor(N/2); end

[bInd, bFreq]= getBandIndices(band, dat.fs, N);
XX= zeros(1, nChans*nEvents);
nWindows= 1 + max(0, floor((T-N)/opt.step));
iv= 1:min(N, T);
Win= repmat(opt.win(:), [1 nChans*nEvents]);

for iw= 1:nWindows,
  Xf= fft(dat.x(iv,:).*Win, N);
  XX= XX + mean( abs(Xf(bInd,:)).^2, 1 );
  iv= iv + opt.step;
end
XX = XX/(nWindows*normWin^2);
dat.x= reshape(10*log10(XX+eps), [1 nChans nEvents]);
dat.t= dat.t(end);
dat.yUnit= 'dB';
