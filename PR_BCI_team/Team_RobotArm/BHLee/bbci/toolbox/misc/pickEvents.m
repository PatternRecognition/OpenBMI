function [mrk, ev]= pickEvents(mrk, ev)
%mrk= pickEvents(mrk, ev)
%[mrk, ev]= pickEvents(mrk, nEvents)
%
% pick events with indices ev or
% pick randomly nEvents events from the marker structure
% (if it contains more than maxEvents events)

if ischar(ev), ev= {ev}; end
if iscell(ev),
%  clInd= find(ismember(mrk.className, ev));
%% the command above would not keep the order of the classes in cell 'ev'  
  warning('obsolete usage: use mrk_selectClasses in this case');
  clInd= [];
  for desiredClass= ev,
    clInd= [clInd find(ismember(mrk.className, desiredClass))];
  end
  ev= find(any(mrk.y(clInd,:),1));
end

if length(ev)==1,
  maxEvents= ev;
  nEvents= length(mrk.toe);
  ev= randperm(nEvents);
  if nEvents>maxEvents,
    ev= ev(1:maxEvents);
  end
end

mrk.pos= mrk.pos(ev);
mrk.toe= mrk.toe(ev);
if isfield(mrk, 'trg'),
  mrk.trg= mrk.trg(ev);
end

if isfield(mrk, 'y'),
  mrk.y= mrk.y(:,ev);
  nonvoidClasses= find(any(mrk.y,2));
  if length(nonvoidClasses)<size(mrk.y,1),
    warning(sprintf('void classes removed, %d classes remaining', ...
                    length(nonvoidClasses)));
    mrk.y= mrk.y(nonvoidClasses,:);
    if isfield(mrk, 'className'),
      mrk.className= {mrk.className{nonvoidClasses}};
    end
  end
end
