function out= proc_joinChannels(dat, join, varargin)
%out= proc_joinChannels(dat, channel_list, <remainChans>)
%
% e.g. channel_list= {'C4+C2', CP4+CP2'}
%
% was written for a special purpose, is not of general interest

% bb, ida.first.fhg.de


if ~iscell(join), join={join}; end

out= copyStruct(dat, 'x','clab');
[T nChans, nEpos]= size(dat.x);
nJoins= length(join);
out.x= zeros(T, nJoins, nEpos);
out.clab= cell(1, nJoins);
for ij= 1:nJoins,
  joStr= join{ij};
  is= find(joStr=='+' | joStr=='-');
  if isempty(is),
    ch= chanind(dat, joStr);
    out.x(:,ij,:)= dat.x(:,ch,:);
    out.clab{ij}= dat.clab{ch};
    continue;
  end
  is= [is length(joStr)+1];
  for kk= 1:length(is)-1,
    joFcn= joStr(is(kk));
    ch1= chanind(dat, joStr(1:is(kk)-1));
    ch2= chanind(dat, joStr(is(kk)+1:is(kk+1)-1));
    out.x(:,ij,:)= eval(['dat.x(:,ch1,:)' joFcn 'dat.x(:,ch2,:)']);
    out.clab{ij}= [dat.clab{ch1} joFcn dat.clab{ch2}];
  end
end

out= proc_copyChannels(out, dat, varargin);
