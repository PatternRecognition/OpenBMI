function mrk= getPacedEvents(file, jit_tol, classDef, blockingDoubles, ...
                             varargin)
%mrk= getPacedEvents(file, jit_tol, classDef, <blockingDoubles=55, fs=100>)

if ~exist('jit_tol', 'var'), jit_tol=150; end
if ~exist('blockingDoubles', 'var') | isempty(blockingDoubles), 
  blockingDoubles=55; 
end
if ~exist('classDef', 'var') | isempty(classDef), 
  classDef= {'F', 'J'; 'left index', 'right index'};
end

mrk= readMarkerTable(file, varargin{:});

iStimulus= find(mrk.toe==1);
iAllKeys= find(ismember(mrk.toe, [classDef{1,:}]));

nEvents= length(iStimulus);
iKey= zeros(1, nEvents);
doubleHit= [];
for ei= 1:nEvents,
  [dummy,ik]= min(abs( mrk.pos(iAllKeys) - mrk.pos(iStimulus(ei)) ));
  if ik<length(iAllKeys),
    diff= (mrk.pos(iAllKeys(ik+1)) - mrk.pos(iAllKeys(ik))) / mrk.fs*1000;
    if diff<blockingDoubles,
      doubleHit= [doubleHit  ei];
    end
  end
  iKey(ei)= iAllKeys(ik);
end
mrk.jit= (mrk.pos(iKey) - mrk.pos(iStimulus)) / mrk.fs*1000;
mrk.trg= mrk.pos(iStimulus);
mrk.pos= mrk.pos(iKey);
mrk.toe= mrk.toe(iKey);

offTrigger= find( abs(mrk.jit) > jit_tol );
reject= [doubleHit offTrigger];
accept= setdiff(1:nEvents, reject);
mrk.toe= mrk.toe(accept);
mrk.pos= mrk.pos(accept);
mrk.jit= mrk.jit(accept);
mrk.trg= mrk.trg(accept);


nClasses= size(classDef,2);
mrk.y= zeros(nClasses, length(mrk.toe));
for ic= 1:nClasses,
  mrk.y(ic,:)= ismember(mrk.toe, classDef{1,ic});
end
if size(classDef,1)>1,
  mrk.className= {classDef{2,:}};
end


fprintf('rejected: %d (off trigger),  %d (double hit)\n', ...
        length(offTrigger), length(doubleHit));
