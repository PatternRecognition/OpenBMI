function [varargout] = phaseAmplitudeEstimate(dat, f, param) 
% [phase, amplitude, W_tf] = phaseAmplitudeEstimate(dat, f, param) 
% alternatively You can call  
% [W_tf] = phaseAmplitudeEstimate(dat, f, param) 
% 
% IN:
%   dat   - data struct ('cnt' or 'epo')
%   f     - center frequency of the wavelet
%   param - optinal struct with fields
%        .omega_0  -  the parameter for the mother morlet wavelet
%        .eFolding -  controls the extension of the
%                     wavelet in the time domain, is used as a 
%                     multiplier to the e-folding time of the wavelet 
%
% OUT:  
%   W_tf - complex wavelet transform of the real signal

  if ~exist('param','var') ,
    param = [] ;
  end ;
  if nargout == 1,
    varargout{1} = waveletTransform(dat, f, param) ;
  elseif nargout <= 3,
    W_tf = waveletTransform(dat, f, param);
    
    % calculate the instantaneous amplitude and phase
    varargout{1} = copy_struct(dat, 'not', 'x') ;
    varargout{2} = varargout{1} ;
  
    varargout{2}.x = abs(W_tf.x) ;
    varargout{1}.x = angle(W_tf.x) ;
    if nargout == 3,
      varargout{3}= W_tf ;
    end
  else 
    error('wrong number of output arguments in phaseAmplitudeEstimate.m');
  end ;

  