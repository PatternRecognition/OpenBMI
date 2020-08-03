function mrk= getPacedSemiimagEvents(file, tol, classDef, blockingDoubles, ...
                             varargin)
%mrk= getPacedSemiimagEvents(file, tol, classDef, <blockingDoubles=55, fs=100>)
%
% e.g. classDef= {-['A','F'], []; 'left real', 'right imag'}

if ~exist('tol', 'var'), tol=500; end
if ~exist('blockingDoubles', 'var') | isempty(blockingDoubles), 
  blockingDoubles=55; 
end
if size(classDef,2)~=2, error('only two classes allowed'); end
imagClass= find([isempty(classDef{1,1}) isempty(classDef{1,2})]);
if isempty(imagClass), error('imag class must be void'); end

imagMrk= 1;

Mrk= readMarkerTable(file, varargin{:});

iStimulus= find(Mrk.toe==1);
iAllKeys= find(ismember(Mrk.toe, [classDef{1,:}]));
iBlockStart= find(Mrk.toe==252);
for ib= 1:length(iBlockStart),
  iFirstStimInBlock(ib)= min(find(iStimulus>iBlockStart(ib)));
end

nEvents= length(iStimulus);
mrk.pos= zeros(1, nEvents);
mrk.toe= zeros(1, nEvents);
mrk.trg= zeros(1, nEvents);
reject= [];
%% each block starts with the first real keypress
%% stimuli before are just to get into the rhythm
for ei= 1:nEvents,
  if ismember(ei, iFirstStimInBlock),
    firstHitEncountered= 0;
  end
  [absjit,ik]= min(abs( Mrk.pos(iAllKeys) - Mrk.pos(iStimulus(ei)) ));
  absjitMsec=  absjit/Mrk.fs*1000;
  if ~firstHitEncountered,
    if absjitMsec < tol,
      firstHitEncountered= 1;
    else
      reject= [reject ei];
      continue;
    end
  end
  if absjitMsec < tol,
    if ik<length(iAllKeys),
      diff= (Mrk.pos(iAllKeys(ik+1)) - Mrk.pos(iAllKeys(ik))) / Mrk.fs*1000;
      if diff<blockingDoubles,
        reject= [reject  ei];
      end
    end
    iKey= iAllKeys(ik);
    mrk.pos(ei)= Mrk.pos(iKey);
    mrk.toe(ei)= Mrk.toe(iKey);
    mrk.trg(ei)= Mrk.pos(iStimulus(ei));
  else
    mrk.pos(ei)= Mrk.pos(iStimulus(ei));
    mrk.toe(ei)= imagMrk;
    mrk.trg(ei)= Mrk.pos(iStimulus(ei));
  end
end

accept= setdiff(1:nEvents, reject);
mrk.toe= mrk.toe(accept);
mrk.pos= mrk.pos(accept);
mrk.trg= mrk.trg(accept);
mrk.fs= Mrk.fs;

mrk.y= repmat(ismember(mrk.toe, classDef{1,3-imagClass}), [2 1]);
mrk.y(imagClass,:)= 1 - mrk.y(imagClass,:);
if size(classDef,1)>1,
  mrk.className= {classDef{2,:}};
end
