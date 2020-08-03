function mrk= getTriggered_cncrEvents(file, classDef, tol, blockingDoubles, ...
                                      varargin)
%mrk= getTriggered_cncrEvents(file, classDef, tol=200, ...
%                             <blockingDoubles=55, fs=100>)
%
% e.g. classDef= {10, 11, 12; ...
%                 -'F', [], []; ...
%                 'left click', 'right no-click', 'rest'}

if ~exist('tol', 'var') | isempty(tol), tol=200; end
if ~exist('blockingDoubles', 'var') | isempty(blockingDoubles), 
  blockingDoubles=55; 
end

marker_pace= 1;  %% beat of pace maker is marked 1 (BV marker S  1)
marker_stimuli= [classDef{1,:}];
iClickClass= 1;
while isempty(classDef{2,iClickClass}),
    iClickClass= iClickClass+1;
end

Mrk= readMarkerTable(file, varargin{:});

iPace= find(Mrk.toe==marker_pace);
iStimulus= find(ismember(Mrk.toe, marker_stimuli));
iAllKeys= find(ismember(Mrk.toe, [classDef{2,:}]));

nEvents= length(iStimulus);
mrk.pos= zeros(1, nEvents);
mrk.toe= zeros(1, nEvents);
mrk.latency= NaN*zeros(1, nEvents);
mrk.indexedByEpochs= {'latency'};
reject= [];
ip= 0;
ep= length(iPace);
for ei= 1:nEvents,
  is= iStimulus(ei);
  mrk.toe(ei)= Mrk.toe(is);

  if ip==ep,
    warning('missing pace marker after stimulus');
    reject= [reject, ei:nEvents];
    break;
  end
  pp= min( find(Mrk.pos(iPace(ip+1:ep)) > Mrk.pos(is)) );
  ip= ip+pp;
  mrk.pos(ei)= Mrk.pos(iPace(ip));

  if Mrk.toe(is)==classDef{1,iClickClass},
    [absjit,ik]= min(abs( Mrk.pos(iAllKeys) - mrk.pos(ei) ));
    absjitMsec=  absjit/Mrk.fs*1000;
    if absjitMsec < tol,
      if ik<length(iAllKeys),
        diff= (Mrk.pos(iAllKeys(ik+1)) - Mrk.pos(iAllKeys(ik))) / Mrk.fs*1000;
        if diff<blockingDoubles,
          reject= [reject, ei];
        end
      end
      iKey= iAllKeys(ik);
%%      mrk.toe(ei)= Mrk.toe(iKey);
      mrk.latency(ei)= Mrk.pos(iKey) - mrk.pos(ei);
    else
      warning('missing click after click stimulus');
      reject= [reject, ei];
    end
  else
    [absjit,ik]= min(abs( Mrk.pos(iAllKeys) - mrk.pos(ei) ));
    absjitMsec=  absjit/Mrk.fs*1000;
    if absjitMsec < tol,
      warning('click after no-click (or rest) stimulus');
      reject= [reject, ei];
    end
  end
end

accept= setdiff(1:nEvents, reject);
mrk.toe= mrk.toe(accept);
mrk.pos= mrk.pos(accept);
mrk.latency= mrk.latency(accept);
mrk.fs= Mrk.fs;

nClasses= length(marker_stimuli);
mrk.y= zeros(nClasses, length(accept));
for ic= 1:nClasses,
  mrk.y(ic,:)= (mrk.toe==marker_stimuli(ic));
end
if size(classDef,2)>2,
  mrk.className= {classDef{3,:}};
end
