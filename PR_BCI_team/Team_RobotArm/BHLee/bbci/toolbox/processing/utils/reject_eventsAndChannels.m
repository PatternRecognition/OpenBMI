function [mrk, rClab, rTrials]= ...
    reject_eventsAndChannels(cnt, mrk, ival, varargin)
%REJECT_EVENTSANDCHANNELS - Artifact rejection for events and channels
%
%Synopsis:
% [mrk, rClab, rTrials]= ...
%    reject_eventsAndChannels(CNT, MRK, IVAL, ...)
%
%Arguments:
% CNT: data structure of continuous signals
% MRK: event marker structure
% IVAL: time interval (relative to marker events) which is to be checked
% optional properties:
%  .do_bandpass:  default 1.
%  .band       :  default [5 40]
%  .clab       :  {'not', 'E*'}
%
%Returns:
% MRK: marker structure in which rejected trials are discarded
% rClab: cell array of channel labels of rejected channels
% rTrials: indices of rejected trials

% Author(s): Benjamin Blankertz, Aug 2006


opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'do_bandpass', 1, ...
                  'band', [5 40], ...
                  'clab', {'not','E*'});

cnt= proc_selectChannels(cnt, opt.clab);

if opt.do_bandpass,
  [b,a]= butter(5, opt.band/cnt.fs*2);
  cnt= proc_filt(cnt, b, a);
end

fv= makeEpochs(cnt, mrk, ival);
fv= proc_variance(fv);
V= squeeze(fv.x);

nChans= length(cnt.clab);
nEvents= size(fv.y,2);
chGood= [1:nChans]';
evGood= 1:nEvents;
if ~isequal(size(V), [nChans nEvents]),
  error('dimension confusion');
end


%% first-pass channels: remove channels with variance droppping to zero
%%  criterium: variance<1 in more than 5% of trials
dropClab= find(mean(V<1,2) > .05);

V(dropClab,:)= [];
rClab= dropClab;
chGood(dropClab)= [];


%% first-pass trials: remove really bad trials
%%  criterium: >= 20% of the channels have excessive variance
Vm= median(V, 1);
Vdm= median(abs(V-repmat(Vm, [length(chGood) 1])), 1);
EX= ( V > repmat(Vm+10*Vdm, [length(chGood) 1]) );
rTrials= find( mean(EX,1)>0.2 );

V(:,rTrials)= [];
evGood(rTrials)= [];


%% second-pass channels: reject channels with excessive variance
%%   TODO: respect topography - relative to neighbors (?)
Vm= median(V, 2);
Vmm= median(Vm);
Vmdm= median(abs(Vm-repmat(Vmm, [length(chGood) 1])));
exClab= find(Vm > Vmm+10*Vmdm);

V(exClab,:)= [];
rClab= [rClab; chGood(exClab)];
chGood(exClab)= [];


%% second-pass trials
Vm= median(V, 1);
Vmm= median(Vm);
Vmdm= median(abs(Vm-repmat(Vmm, [1 length(evGood)])));
rTr= find(Vm > Vmm+5*Vmdm);

V(:,rTr)= [];
rTrials= [rTrials evGood(rTr)];
evGood(rTr)= [];


%% third-pass channels
Vv= var(V')';
rC= find(Vv > 10*median(Vv));

V(rC,:)= [];
rClab= [rClab; chGood(rC)];
chGood(rC)= [];


%% third-pass trials
%%  remove trials with variance peaks in single channels
Vm= median(V(:));
Vdm= median(abs(V(:)-Vm));
rTr= find(any(V > Vm+20*Vdm, 1));

V(:,rTr)= [];
rTrials= [rTrials evGood(rTr)];
evGood(rTr)= [];


rClab= cnt.clab(rClab);

mrk= mrk_selectEvents(mrk, setdiff(1:nEvents, rTrials));
