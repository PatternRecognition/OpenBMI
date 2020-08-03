function dat= proc_waveletPhase(dat, freq, varargin);

dat= singleWavelet(dat, freq, varargin{:});
dat.x= angle(dat.x);
