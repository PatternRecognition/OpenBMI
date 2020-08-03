function mrk= fixMultiProblem(mrk)
%mrk= fixMultiProblem(mrk)

nEvents= length(mrk.toe);
valid= ones(1, nEvents);
for im= 1:nEvents,
  if mrk.toe(im)==78,
    if im>1 & (mrk.pos(im)-mrk.pos(im-1))/mrk.fs*1000<150,
      valid(im)= 0;
    else
      mrk.toe(im)= 70;
    end
  end
end
mrk.pos= mrk.pos(find(valid));
mrk.toe= mrk.toe(find(valid));
