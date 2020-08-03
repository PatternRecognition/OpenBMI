function mrk = combineMarkerToken(mrk,tim);
%COMBINEMARKERTOKEN COMBINES ALL EQUAL TOKEN IN A SPECIFIED TIME WINDOW
%
% usage:
% mrk = combineMarkerToken(mrk,tim);
%
% input:
% mrk     a marker structure
% tim     time in msec markers are joined together (to the first one)
%
% output:
% mrk     a reduced mrk structure
%
% Guido Dornhege, 02/09/04

tim = tim/1000*mrk.fs;

ind = [];
i = 0;
while i<length(mrk.pos)
  i = i+1;
  ind = [ind,i];
  po = mrk.pos(i);
  while i<length(mrk.pos) & mrk.toe(i)==mrk.toe(i+1) & mrk.pos(i+1)<po+tim
    i = i+1;
  end
end


mrk = mrk_selectEvents(mrk,ind);

  