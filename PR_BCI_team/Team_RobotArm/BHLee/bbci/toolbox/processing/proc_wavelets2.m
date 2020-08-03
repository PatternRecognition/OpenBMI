function dat = proc_wavelets(dat,varargin)

% PROC_WAVELETS -  calculates the continuous wavelet transform for a
% specified range of scales. Coefficients are obtained by convolution of 
% the EEG data with a (possibly complex) wavelet.
%
%Usage:
% dat = proc_wavelets(dat,<OPT>)
% dat = proc_wavelets(dat,freq,<OPT>)
%
%Arguments:
% DAT      -  data structure of continuous or epoched data (2D or 3D)
% OPT - struct or property/value list of optional properties:
% 'mother' -  mother wavelet (default 'morlet'). Morlet is currently the
%             only implemented wavelet.
% 'freq'   -  frequencies to be considered (default: alpha band [8:12])
% 'res' - 
% 'support' - since wavelets such as Morlet do not have compact 
%             support (ie, they are not zero at some point), they have to be cut off
%             at some point for filtering purposes. Support is given as
%             times the e-folding time, that is the time after which the
%             wavelet power falls off by e^-2. (default 2)
% 'norm' -    if set to 'unit', all time-frequency bins are
%             scaled to unit power and directly comparable. If set to
%             'amplitude', the amplitude directly reflects the amplitude in
%             the signal. (default 'unit')
% 'w0'    -   unitless frequency constant defining the trade-off between
%             frequency resolution and time resolution. For Morlet
%             wavelets, it sets the width of the Gaussian window. (default 6)
% 'vectorize' - vectorization is used for faster processing at the cost of
%             higher memory consumption (default 1)
%
%Returns:
% DAT    -    updated data structure with a higher dimension (3D or 4D).
%             For continuous data, the dimensions correspond to frequency x
%             time x channels. For epoched data, frequency x time x
%             channels x epochs.
% Furthermore, the following fields are added to DAT:
% .wave_length    - length of each wavelet in time samples
% .wave_freq      - wavelet center frequencies
% .wave_fun       - wavelet functions
%
% Interpretation of the data: Wavelet coefficients are complex, that is,
% consisting of a real part real(dat.x) and an imaginary part imag(dat.x). 
% Use abs(dat.x) and angle(dat.x) to get amplitude and phase spectra.
%
%Memory consumption: The dimensionality of the data is increased by one dimension,
% leading to a substantial increase in memory consumption. Selection
% of a small subset of electrodes and trials is strongly recommended.
%
% See also PROC_SPECTROGRAM.

% Author: Matthias Treder (2010)

clear i norm psi

if numel(varargin)>0 && ~ischar(varargin{1})
  freq = varargin{1};
  varargin(1) = [];
else
  freq = 8:12;
end

opt= propertylist2struct(varargin{:});
[opt, isdefault]= ...
    set_defaults(opt, ...
                 'mother','morlet', ...
                 'w0',7,...
                 'freq', freq, ...
                 'vectorize', 1, ...
                 'support', 2);


dt = 1/dat.fs;  % time resolution (determined by sampling frequency)
norm = [];  % Normalization 
w0 = opt.w0;
e = exp(1);  % Euler's number

%% Define mother wavelet, normalization factor, and fourier period
% [ and normalization ..]
switch(opt.mother)
  case 'morlet',
    % Morlet wavelet normalized to unit energy
    % includes the correction factor e.^(-w0^2/2) which becomes
    % significant only for w0 <= 4
    psi = 'pi^(-1/4) .* (e.^(i*w0*t) - e.^(-w0^2/2) ).* e.^(-t.^2/2)';
    norm = 'sqrt(dt/s)';  % Normalization factor for each child wavelet
    scf = '(w0+sqrt(2+w0^2))/(4*pi*f)'; % Scale as a function of Fourier frequency  
    efold = 'sqrt(2)*s'; % E-folding time
  otherwise
    error(['Wavelet ' opt.mother ' not implemented.'])
end

%% Define scales corresponding to the desired frequencies
scales = [];    % Wavelet scales
for f=opt.freq
  scales = [scales eval(scf)];
end

%% New fields
dat.wave_freq = opt.freq;
dat.wave_width = [];  % 
dat.wave_freq_res = []; % Frequency resolution of wavelet (=standard deviation in freq domain)
dat.wave_time_res = []; % Time resolution of wavelet (=standard deviation in time domain)
dat.wave_fun = {};  % Wavelet functions

xx = zeros([length(opt.freq) size(dat.x)]);  % first dimension is now frequencies

%% Traverse scales
snum = 1;
for s=scales
  
  %% Calculate e-folding time for current scale
  ef = eval(efold);
  
  %% Calculate time samples
  t = 0:dt:ef*opt.support;
  t = [-fliplr(t) t(2:end)]; % Add also negative part
  t = t/s;   % Scale time
  
  %% Evaluate mother wavelet
  wave = eval(psi);

  %% Calculate normalization factor
  eval(['normfac = ' norm ';'])

  wave = wave * normfac;
  wave = wave - mean(wave(:)); %% Set wavelet to DC = 0
  
  dat.wave_width = [dat.wave_width numel(wave)];
%   dat.wave_freq_res = [dat.wave_freq_res 1/(2*pi*sigmat)];
%   dat.wave_time_res = [dat.wave_time_res sigmat];
  dat.wave_fun = {dat.wave_fun{:} wave};
  
  %% Conjugate and mirror (because the filter function mirrors the filter)
  wave = conj(fliplr(wave));
  
  %% The "filter" function implements a causal filter. To obtain a
  %% non-causal filter results need to be shifted. Therefore, also append 
  %% zeros to the end 
  
  halfwid = ceil(length(wave)/2);
  if opt.vectorize
      %% (vectorization by haufe)
      appzero = repmat(zeros(size(mean(dat.x, 1))), [floor(length(wave)/2), 1, 1]);
      out = filter(wave, 1, cat(1, dat.x, appzero));
      xx(snum, :, :, :) = out(halfwid:end, :, :, :);
  else
      %% Proceed through all channels and epochs
      appzero = zeros(floor(length(wave)/2), 1);
      for cc=1:size(dat.x,2)
        if ndims(dat.x)==2
          out = filter(wave,1,squeeze([dat.x(:,cc); appzero]));
          % Shift to get non-causal filter
          out = out(halfwid:end);
          xx(snum,:,cc) = out;
        elseif ndims(dat.x)==3
          for ep=1:size(dat.x,3)
            out = filter(wave,1,squeeze([dat.x(:,cc,ep); appzero]));
            % Shift to get non-causal filter
            out = out(halfwid:end);
            xx(snum,:,cc,ep) = out;
          end
        else error('dat.x must have 2 or 3 dimensions')
        end
      end
  end
  snum= snum+1;
end

%% Replace dat.x with timeXfrequency wavelet coefficients 
dat.x = xx;

