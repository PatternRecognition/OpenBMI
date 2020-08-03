function [W_tf] = waveletTransform(dat, f, param)
% [W_tf] = waveletTransform(dat, f, <param>)
% 
% IN:
%   dat   - data struct ('cnt' or 'epo')
%   f     - center frequency of the wavelet
%   param - optinal struct with fields
%        .omega_0  -  the parameter for the mother morlet wavelet
%        .eFolding -  adjusts the extension of the
%                     wavelet in the time domain, is used as a 
%                     multiplier to the e-folding time of the wavelet 
%
% OUT:  
%   W_tf - complex wavelet transform of the real signal

  if ~exist('param','var') ,
    param = [] ;
  end ;

  if ~isfield(param,'omega_0') ,
    omega_0 = 6;
  else 
    omega_0 = param.omega_0 ;
  end ;

  if ~isfield(param,'times_eFolding') ,
    eFolding = 2 ;
  else 
    eFolding = param.eFolding ;
  end ;
  
  
  % get all dimensionalities
  [T, nChans, nEvents] = size(dat.x) ;

  % create structures to store the transformed signal
  W_tf  = copyStruct(dat) ;

  % get wavelet filter coefficients for frequency 'f'
  [W_tf.filterCoeff, W_tf.t_eff, W_tf.f_eff, W_tf.scale] = gaborWin(omega_0, eFolding, f, dat.fs) ;
  W_tf.filterCoeff = conj(W_tf.filterCoeff) ; 
  filterHalf = (length(W_tf.filterCoeff) -1)/2 ;
  
  % wavelet transform the signal at frequence 'f'
  for ch = 1:nChans,
    dummy = filter(W_tf.filterCoeff(end:-1:1), 1, [dat.x(:,ch,:); ...
		    zeros(filterHalf, 1, nEvents)]);
    dummy(1:filterHalf,:,:) = [] ; 
    W_tf.x(:,ch,:) = dummy ;
  end ;
  
  W_tf.f = f ;


function [win, t_eff, f_eff, scaling] = gaborWin(omega_0, timeHorizont, f_x, fs)
% returns window in the time domain with a morlet wavelet,
% centered at the frequency 'f_x', given the sampling frequency 'fs'
% windowlength is four times the e-folding time 't_eff' 
% in the frequency domain 'f_eff' gives the effective bandwith of
% the wavelet.


  % wavelength of the center-frequency f_x w.r.t. sampling freq. fs
  wavelength_x = fs/f_x ;          % in [samples]
  
  %scaling factor for the morlet wavelet
  scaling = wavelength_x /(4*pi) * (omega_0 + sqrt(2+omega_0^2)) ;
  
  % e-folding time in [samples]
  t_eff = sqrt(2) * scaling ;
  t_eff_INT = ceil(t_eff) ; % integer value

  % e-folding time in [ms]
  t_eff = t_eff *1000/fs ;
  
  % e-folding frequency
  delta_f = f_x - omega_0 ;
  f_eff = sqrt(2)/scaling *fs/2/pi;
  
  % time window 
  windowHalf = ceil(timeHorizont*t_eff_INT) ;
  t = -windowHalf:windowHalf ;
  
  % rescale the time window
  t = t/scaling ;
  
  % morlet wavelet at rescaled time
%  win = 1/sqrt(scaling)/(pi^.25) * exp(i*omega_0*t) .* exp(-t.^2/2) ;

  % aternatively the last line might be replace by the following
  % four lines
  
    win = 1/(pi^.25) * exp(i*omega_0*t) .* exp(-t.^2/2) ;
    normalization = sum(abs(win).^2) ;
    win = win./sqrt(normalization) ;
    scaling = normalization ;
