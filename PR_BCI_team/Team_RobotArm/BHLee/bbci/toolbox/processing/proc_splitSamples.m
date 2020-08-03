% [epotr, epote, iTr, iTe]=proc_splitSmaples(epo, nFold)
% [epotr, epote, iTr, iTe]=proc_splitSmaples(epo, {iTr, iTe})
function [epotr, epote, iTr, iTe]=proc_splitSmaples(epo, nFold)

if iscell(nFold)
  iTr = nFold{1};
  iTe = nFold{2};
else
  if isequal(nFold, [1 1])       % chronological validation
    n=size(epo.y,2); h=ceil(n/2);
    iTr = 1:h;
    iTe = h+1:n;
  else
    [divTr divTe] = sample_kfold(epo.y, [1 nFold]);

    iTr = divTr{:}{1};
    iTe = divTe{:}{1};
  end
end

epotr = proc_selectEpochs(epo, iTr);
epote = proc_selectEpochs(epo, iTe);
