function mrk= mrk_removeEmptyRuns(mrk)
%MRK_REMOVEEMPTYRUNS - Remove run numbers of empty runs
%
%This functions should be superficial.

nRuns= max(mrk.run_no);
iEmpty= find(~ismember(1:nRuns, mrk.run_no));
if ~isempty(iEmpty),
  for ii= iEmpty;
    for jj= ii+1:nRuns,
      idx= find(mrk.run_no==jj);
      mrk.run_no(idx)= jj-1;
    end
  end
end
