function dat= proc_bandPower(dat, varargin)
%dat= proc_bandPower(dat, band, N, step)
%
% calculate band energy from spectral density. for short time
% epochs rather use proc_fourierBandEnergy
%
% SEE proc_spectrum, proc_fourierBandEnergy

dat= proc_spectrum(dat, varargin{:});
dat.x= sum(dat.x);
