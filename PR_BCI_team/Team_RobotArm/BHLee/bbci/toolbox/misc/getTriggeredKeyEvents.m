function mrk= getTriggeredKeyEvents(file, trigDef, jit_tol, ...
                                    blockingDoubles, varargin)
%mrk= getTiggeredEvents(file, trigDef, ...
%                       <jit_tol=250, blockingDoubles=55, fs=100>)
%
% IN:
%      jit_tol - jitter tolerance [ms], default 250
%      blocking doubles - if two keys are pressed within a smaller interval
%                         than this, the respective trial is rejected

GO_MARKER= 1;

if ~exist('jit_tol', 'var'), jit_tol=250; end
if ~exist('blockingDoubles', 'var') | isempty(blockingDoubles), 
  blockingDoubles=55; 
end

mrk= readMarkerTable(file, varargin{:});

iAllGo= find(mrk.toe==GO_MARKER);
iStimulus= find(ismember(mrk.toe, [trigDef{1,:}]));
iAllKeys= find(ismember(mrk.toe, [trigDef{2,:}]));

iStimulus(find(iStimulus>max(iAllGo)))= [];
nEvents= length(iStimulus);
iGo= zeros(1, nEvents);
iKey= zeros(1, nEvents);
doubleHit= [];
for ei= 1:nEvents,
  ig= min( find(iAllGo > iStimulus(ei)) );
  iGo(ei)= iAllGo(ig);
  ik= min( find(iAllKeys > iStimulus(ei)) );
  if ik<length(iAllKeys),
    diff= (mrk.pos(iAllKeys(ik+1)) - mrk.pos(iAllKeys(ik))) / mrk.fs*1000;
    if diff<blockingDoubles,
      doubleHit= [doubleHit  ei];
    end
  end
  iKey(ei)= iAllKeys(ik);
end


mrk.jit= (mrk.pos(iKey) - mrk.pos(iGo)) / mrk.fs*1000;
mrk.go= mrk.pos(iGo);
mrk.trg= mrk.pos(iStimulus);
mrk.pos= mrk.pos(iKey);
mrk.toe= mrk.toe(iStimulus);

offTrigger= find( abs(mrk.jit) > jit_tol );
reject= [offTrigger doubleHit];
accept= setdiff(1:nEvents, reject);
mrk.toe= mrk.toe(accept);
mrk.pos= mrk.pos(accept);
mrk.jit= mrk.jit(accept);
mrk.trg= mrk.trg(accept);
mrk.go= mrk.go(accept);


nClasses= size(trigDef,2);
mrk.y= zeros(nClasses, length(mrk.toe));
for ic= 1:nClasses,
  mrk.y(ic,:)= ismember(mrk.toe, trigDef{1,ic});
end
if size(trigDef,1)>2,
  mrk.className= {trigDef{3,:}};
end


fprintf('rejected: %d (off trigger),  %d (double hit)\n', ...
        length(offTrigger), length(doubleHit));
