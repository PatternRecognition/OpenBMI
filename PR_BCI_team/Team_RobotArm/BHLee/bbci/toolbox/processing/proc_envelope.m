function dat= proc_envelope(dat, varargin)
%PROC_ENVELOPE - Envelope Curve of Oscillatory Signals
%
%Synopsis:
% DAT= proc_envelope(DAT, <OPT>)
%
%Arguments:
% DAT: data structure, continuous or epoched signals.
% OPT: struct or property/value list of optinal properties
%  .envelop_method: 'hilbert' (only choice so far)
%  .ma_method: 'centered' (default) or 'causal'
%  .ma_msec: window length [msec] for moving average, deafult: 100.
%  .channelwise: useful in case of memory problems
%Output:
% DAT: output data structure, continuous or epoched as input
%      signals are the envelope curves of inut signals

% Author(s): Benjamin Blankertz
% added channelwise option by Claudia

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'envelop_method', 'hilbert', ...
                  'ma_method', 'centered', ...
                  'ma_msec', 100, ...
                  'ma_opts', {}, ...
                  'channelwise', false);

sz= size(dat.x);
switch(lower(opt.envelop_method)),
 case 'hilbert',
     if opt.channelwise
         for ch = 1:sz(2)
             x(:,ch,:) = abs(hilbert(squeeze(dat.x(:,ch,:))));
         end
         dat.x = x;
     else
         dat.x= abs(hilbert(dat.x(:,:)));     
         dat.x= reshape(dat.x, sz);
     end

 otherwise,
  error('unknown envelop method');
end

if ~isempty(opt.ma_opts),
  dat= proc_movingAverage(dat, opt.ma_msec, opt.ma_opts{:});
elseif ~isempty(opt.ma_msec) & opt.ma_msec>0,
  dat= proc_movingAverage(dat, opt.ma_msec, opt.ma_method);
end
