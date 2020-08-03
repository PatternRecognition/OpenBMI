function mrk = mrk_correctResponse(mrk,tim);
%MRK_CORRECTRESPONSE reduced mrk which only correct responses
%
% usage:
% mrk = mrk_correctResponse(mrk,tim);
%
% input:
% mrk      mrk structure with response, .trg with stimulus
% tim      tim (time window afterwards)
%
% output:
% mrk      mrk structure with correct responses

tim = tim/1000*mrk.fs;

ind = [];

for i = 1:length(mrk.trg.pos)
  ti = find(mrk.pos>mrk.trg.pos(i) & mrk.pos<mrk.trg.pos(i)+tim);
  if ~isempty(ti)
    if length(find(sum(mrk.y(:,ti),2)))==1
      if find(mrk.y(:,ti(1)))==find(mrk.trg.y(:,i))
        ind = [ind,[ti(1);i]];
      end
    end
  end
end


mrk.trg = mrk_selectEvents(mrk.trg,ind(2,:));
mrk = mrk_selectEvents(mrk,ind(1,:));

