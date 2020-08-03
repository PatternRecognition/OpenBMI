function dat= proc_delayChannels(dat, tau_ms, clab)
%dat= proc_delayChannels(dat, tau_ms, <clab>)
%
% IN  dat    - struct of continuous or epoched data
%     tau_ms - delay [ms]  (only one value allowed)
%     clab   - channel label (or indices) of which delayed copies
%              are appended, default all.
%
% OUT dat    - struct with delayed channels added

%% bb 07/2004 ida.first.fhg.de


if length(tau_ms)>1, error('tau_ms must be a scalar'); end

[T nChans, nE]= size(dat.x);

if exist('clab','var'),
  ch_idx= chanind(dat, clab);
else
  ch_idx= 1:nChans;
end

tau= round(tau_ms/1000*dat.fs);
if min(tau)<0 | max(tau)>=T,
  error('bad choice of tau');
end

TT= T - max(tau);
non_delay_chans= setdiff(1:nChans, ch_idx);
xx= zeros(TT, nChans, nE);
xx(:, non_delay_chans, :)= dat.x(end-TT+1:end, non_delay_chans, :);
xx(:, ch_idx, :)= dat.x(1:TT, ch_idx, :);

dat.x= xx;
new_clab= strcat(dat.clab(ch_idx), [' lag=' int2str(tau_ms)]);
dat.clab(ch_idx)= new_clab;
if isfield(dat, 't'),
  dat.t= dat.t(end-TT+1:end);
end
