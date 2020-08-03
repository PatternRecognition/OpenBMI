if isequal(test_file, train_file),
  nTrains= floor(length(mrk.pos)*train_frac);
  test_begin= mrk.pos(nTrains) + train_test_delay/1000*mrk.fs;
else
  nTrains= 0;
  test_begin= max(1, mrk.pos(1) - 2*mrk.fs);
end

%% end of testing interval
all_mrk= readMarkerTable(cnt.title);
fin_marker= max(find(all_mrk.toe==253));
if isempty(fin_marker), 
  test_end= mrk.pos(end) + mrk.fs;    %% stop 1 sec after last regular event
else
  test_end= all_mrk.pos(fin_marker);  %% stop at end marker (#253)
end

if ~isequal(FB_TYPE, 'none'),
  fprintf('press <ret> to start feedback\n'); pause
end
