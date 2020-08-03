function [b, a]= procutil_firlsFilter(freq, fs, varargin)
%PROCUTIL_FIRLSFILTER - Get coefficients of an FIR-ls filter
%
%The algorithm to determine the settings was taken from the function
%eegfilt of the eeglab toolbox by Scott Makeig and Arnaud Delorme.
%
%Synopsis:
%  [B, A]= procutil_firlsFilter(FREQ, FS, <OPT>)
%
%Arguments:
%  FREQ - Determines the edge frequency/cies of the filter. If
%     FREQ is a two-element vector, a band-pass filter is designed.
%     If FREQ is a scalar, by default a high-pass filter is designed.
%  FS - Sampling rate
%  OPT - Struct or property/value list of optional properties:
%    .lowpass
%
%Returns:
%  A = 1 and
%  B: Filter coefficients of the FIR filter. Can be used with the
%    matlab filtering functions filter and filtfilt, as well as
%    with proc_filt, proc_filtfilt
%
%Advice by Scott: When a bandpass filter is unstable, first highpass,
%  then lowpass filtering the data may work.


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'lowpass', 0, ...
                  'bandstop', 0, ...
                  'order', [], ...
                  'minorder', 15, ...
                  'trans', 0.15);

nyq= fs/2;
if isempty(opt.order),
  opt.order= max(opt.minorder, 3*fix(fs/min(freq)));
end

if length(freq)==1 && ~opt.lowpass,
  f= [0 freq*(1-opt.trans)/nyq freq/nyq 1];
  amp= [0 0 1 1];
elseif length(freq)==1 && opt.lowpass,
  f= [0 freq/nyq freq*(1+opt.trans)/nyq 1];
  amp= [1 1 0 0];
elseif length(freq)==2 && ~opt.bandstop,
  f= [0 freq(1)*(1-opt.trans)/nyq freq(1)/nyq ...
      freq(2)/nyq freq(2)*(1+opt.trans)/nyq 1];
  amp= [0 0 1 1 0 0];
elseif length(freq)==2 && opt.bandstop,
  f= [0 freq(1)*(1-opt.trans)/nyq freq(1)/nyq ...
      freq(2)/nyq freq(2)*(1+opt.trans)/nyq 1];
  amp= [1 1 0 0 1 1];
end

b= firls(opt.order, f, amp);
a= 1;
