function [dat,refChans]= online_commonAverageReference(dat, refChans, rerefChans)
%dat= proc_commonAverageReference(dat, <refChans, rerefChans>)
%
% rereference signals to common average reference. you should only
% used scalp electrodes as reference, not e.g. EMG channels.
%
% IN   dat        - data structure of continuous or epoched data
%      refChans   - channels used as average reference, see chanind for format, 
  %                 default scalpChannels(dat)
%      rerefChans - those channels are rereferenced, default refChans
%
% OUT  dat        - updated data structure
%
% SEE scalpChannels, chanind

% bb, ida.first.fhg.de


if nargin<=1 | isempty(refChans),
  refChans= scalpChannels(dat);
end

if ~isstruct(refChans)
    if nargin<=2 | isempty(rerefChans), rerefChans= refChans; end

refChans.rc= chanind(dat, refChans);
refChans.rrc= chanind(dat, rerefChans);
end

car= median(dat.x(:,refChans.rc,:), 2);
%% this might consume too much memory:
%car= repmat(car, [1 length(rrc) 1]);
%dat.x(:,rrc,:)= dat.x(:,rrc,:) - car;

for cc= refChans.rrc,
  dat.x(:,cc,:)= dat.x(:,cc,:) - car;
%  dat.clab{cc}= [dat.clab{cc} ' car'];
end
