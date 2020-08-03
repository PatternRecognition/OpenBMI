function mrk= getTriggeredEvents(file, trigDef, varargin)
%mrk= getTiggeredEvents(file, trigDef, <fs=100>)
%
% IN:
%      lat_tol - latency tolerance [ms], default 750

GO_MARKER= 1;

mrk= readMarkerTable(file, varargin{:});

iAllGo= find(mrk.toe==GO_MARKER);
iStimulus= find(ismember(mrk.toe, [trigDef{1,:}]));

idel= find(iStimulus > iAllGo(end));
if ~isempty(idel),
  iStimulus(idel)= [];
  fprintf('%d trigger deleted (not folowed by go stimulus)\n', length(idel));
end

nEvents= length(iStimulus);
iGo= zeros(1, nEvents);
for ei= 1:nEvents,
  ig= min( find(iAllGo > iStimulus(ei)) );
  iGo(ei)= iAllGo(ig);
end

mrk.trg= mrk.pos(iStimulus);
mrk.toe= mrk.toe(iStimulus);
mrk.pos= mrk.pos(iGo);

nClasses= size(trigDef,2);
mrk.y= zeros(nClasses, length(mrk.toe));
for ic= 1:nClasses,
  mrk.y(ic,:)= ismember(mrk.toe, trigDef{1,ic});
end
if size(trigDef,1)>1,
  mrk.className= {trigDef{2,:}};
end
