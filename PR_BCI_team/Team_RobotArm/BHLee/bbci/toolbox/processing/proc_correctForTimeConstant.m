function out= proc_correctForTimeConstant(dat, time_const, chans)
%dat= proc_correctForTimeConstant(dat, time_const, <chans>)
%
% this function undoes the high-pass filter that was applied by
% the EEG recording device. requires knownledge of the time constant.
% the function should probably only be used for epoched data.
%
% IN   dat        - data structure of continuous or epoched data
%      time_const - time constant (high-pass cut-off)
%                   the was used in recording dat
%      chans      - channels to which correction should be applied
%
% OUT  dat        - updated data structure
%
% thanks to thilo.hinterberger@uni-tuebingen.de

% bb, ida.first.fhg.de


[T, nChans, nEpochs]= size(dat.x);
if ~exist('chans','var'), 
  chans= 1:nChans; 
else
  chans= chanind(dat, chans);
end

f= 1 - exp(-1/(time_const*dat.fs));

out= dat;
for ei= 1:nEpochs,
  d= f * dat.x(:,chans,ei);
  for ti= 1:T,
    out.x(ti,chans,ei)= dat.x(ti,chans,ei) + sum(d(1:ti,:));
  end
end
