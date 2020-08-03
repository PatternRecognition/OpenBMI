function mrk= getDzweiEvents(file, varargin)
%mrk= getDzweiEvents(file, <fs=100>)


Mrk= readMarkerTable(file, varargin{:});
mrk= Mrk;

iStimulus= find(Mrk.toe>=128 & Mrk.toe<=250);
iAllKeys= find(Mrk.toe<128);

nEvents= length(iStimulus);
doubleHit= [];
iKey= zeros(1, nEvents);
mrk.ishit= zeros(1, nEvents);
for ei= 1:nEvents,
  ik= min(find(iAllKeys>iStimulus(ei)));
  if ik<length(iAllKeys),
    diff= (Mrk.pos(iAllKeys(ik+1)) - Mrk.pos(iAllKeys(ik))) / Mrk.fs;
    if diff<1,                     %% next key hit less than one second later?
      doubleHit= [doubleHit  ei];
    end
  end
  iKey(ei)= iAllKeys(ik);
  key= Mrk.toe(iKey(ei));
  stimType= Mrk.toe(iStimulus(ei)) - 128;
  target= bitget(stimType,1)==1 & sum(bitget(stimType,2:5))==2;
  mrk.ishit(ei)= (target & key=='J' ) | (~target & key=='F');
end
mrk.pos= Mrk.pos(iKey);
mrk.toe= Mrk.toe(iKey);
mrk.reac= (Mrk.pos(iKey) - Mrk.pos(iStimulus)) * 1000/Mrk.fs;
mrk.stim= Mrk.pos(iStimulus);

tooSlow= find(mrk.reac>1000);
tooQuick= find(mrk.reac<150);
bad= [doubleHit tooSlow tooQuick];
accept= setdiff(1:nEvents, bad);
mrk.pos= mrk.pos(accept);
mrk.toe= mrk.toe(accept);
mrk.ishit= mrk.ishit(accept);
mrk.reac= mrk.reac(accept);
mrk.stim= mrk.stim(accept);
mrk.y= [mrk.toe=='F'; mrk.toe=='J'];
mrk.className= {'left','right'};
mrk.indexedByEpochs= {'ishit','reac'};

mrk.d2.pos= mrk.pos;
mrk.d2.y= [~mrk.ishit; mrk.ishit];
mrk.d2.className= {'miss', 'hit'};
mrk.d2.fs= mrk.fs;
