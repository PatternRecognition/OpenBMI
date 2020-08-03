function dat = proc_PLF(dat,varargin)
% PROC_PLF -  calculates the phase-locking factor (PLF) given
% time-frequency data. PLF is value between 0 (no phase-locking) and 1
% (perfect phase-locking).
%
%
%Usage:
% dat = proc_PLF(dat,<OPT>)
%
%Arguments:
% DAT      -  data structure of epoched time-frequency data (3D). Values
%             should be complex numbers, so that both phase and magnitude
%             can be extracted.
%                
% OPT - struct or property/value list of optional properties:
%
%Returns:
% DAT    -    a data struct whereby each point in time-frequency space
%             gives the appropriate PLF value.
%
% See also proc_spectrogram proc_wavelets

% Author: Matthias Treder (2010)

clear i 

% opt= propertylist2struct(varargin{:});
% [opt, isdefault]= ...
%     set_defaults(opt, ...
%                  'norm','unit');

% Normalize to unit magnitude
dat.x = dat.x ./ abs(dat.x);

% Classwise average
dat = proc_average(dat);

% Magnitude of the average yiels the PLF
dat.x = abs(dat.x);
