function ival= mrk_extractRunIntervals(mrk)

iStart= find(mrk.toe==252);
nRuns= length(iStart);
ival= [];
for ii= 1:nRuns,
  ie= min(find(mrk.toe==253 & mrk.pos>mrk.pos(iStart(ii))));
  if ~isempty(ie) & (ii==nRuns | mrk.pos(ie)<mrk.pos(iStart(ii+1))),
    ival= [ival; mrk.pos([iStart(ii), ie])];
  end
end
