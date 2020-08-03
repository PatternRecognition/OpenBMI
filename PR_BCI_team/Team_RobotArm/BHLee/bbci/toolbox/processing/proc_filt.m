function dat= proc_filt(dat, varargin)
%PROC_FILT - Digital filtering
%
%Usage:
% DAT= proc_filt(DAT, B, A)
% DAT= proc_filt(DAT, z, p, k)
%
% Apply digital (FIR or IIR) filter. This filtering is causal, but
% produces a phase shift.
%
%Input(1):
% DAT   - data structure of continuous or epoched data
% B,A   - filter coefficients
%
%Input(2):
% DAT   - data structure of continuous or epoched data
% z,p,k - zeros, poles and gain factor
%
%Output:
% DAT   - updated data structure
%
%Example 1:
% % Let cnt be a structure of multi-variate time series ('.x', time along first
% % dimension) with sampling rate specified in field '.fs'.
% [b,a]= butter(5, [7 13]/cnt.fs*2);
% % Apply a causal band-pass filter 7 to 13Hz to cnt:
% cnt_flt= proc_filt(cnt, b, a);
%
%Example 2: Highpass filter above 6 Hz
% % Due to desired high attenuation and small drop-off band (4 Hz to 6 Hz) 
% % and resulting large filter order, the above way of defining a filter via B,A is
% % not possible any more. Use different representation including a gain k:
% db_attenuation = 60;
% Wps = [6 4]/cnt.fs*2;
% [n, Wn]= cheb2ord(Wps(1), Wps(2) , 3, db_attenuation);
% [z,p,k] = cheby2(n, db_attenuation, Wn);        
% cnt_flt= proc_filt(cnt, z,p,k);
%
% 
% Please Note (see cheby2.m): 
% "In general, you should use the [z,p,k] syntax to design IIR filters. 
% For higher order filters (possibly starting as low as order 8),   
% numerical problems due to roundoff errors may occur when forming the 
% transfer function using the [b,a] syntax." 
%
% z,p,k functionality added by David List, Michael Tangermann 26.April 2010
%
% See also proc_filtfilt


if size(varargin,2) == 1
  varargin= cat(2, varargin, {1});
end

if size(varargin,2) == 2
    % varargin(1) == B
    % varargin(2) == A
    dat.x(:,:)= filter(varargin{1}, varargin{2}, dat.x(:,:));
elseif size(varargin,2) == 3 
    % varargin(1) == z
    % varargin(2) == p
    % varargin(3) == k
    [sos_var,g] = zp2sos(varargin{1}, varargin{2}, varargin{3});
    Hd  = dfilt.df2sos(sos_var, g);
    dat.x(:,:) = filter(Hd,dat.x(:,:));
end
