function dat= proc_addDelayedChannels(dat, tau_ms, clab)
%dat= proc_addDelayedChannels(dat, tau_ms, <clab>)
%
% IN  dat    - struct of continuous or epoched data
%     tau_ms - vector of delays [ms]
%     clab   - channel label (or indices) of which delayed copies
%              are appended, default all.
%
% OUT dat    - struct with delayed channels added

%% bb 07/2004 ida.first.fhg.de


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
xx= dat.x(end-TT+1:end,:,:);
for ii= 1:length(tau),
  t= tau(ii);
  xx= cat(2, xx, dat.x(end-TT-t+1:end-t, ch_idx, :));
  new_clab= strcat(dat.clab(ch_idx), [' lag=' int2str(tau_ms(ii))]);
  dat.clab= cat(2, dat.clab, new_clab);
end

dat.x= xx;

if isfield(dat, 't'),
  dat.t= dat.t(end-TT+1:end);
end
