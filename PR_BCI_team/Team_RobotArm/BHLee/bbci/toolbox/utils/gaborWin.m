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
  
  % e-folding frequency
  delta_f = f_x - omega_0 ;
  f_eff = sqrt(2)/scaling *fs/2/pi;
  
  % time window 
  windowHalf = ceil(timeHorizont*t_eff_INT) ;
  t = -windowHalf:windowHalf ;
  
  % rescale the time window
  t = t/scaling ;
  
  % morlet wavelet at rescaled time
  win = 1/sqrt(scaling)/(pi^.25) * exp(i*omega_0*t) .* exp(-t.^2/2) ;

  % aternatively the last line might be replace by the following
  % four lines
  
  %  win = 1/(pi^.25) * exp(i*omega_0*t) .* exp(-t.^2/2) ;
  %  normalization = sum(abs(win).^2) ;
  %  win = win./sqrt(normalization) ;
  %  sacling = normalization ;
  
