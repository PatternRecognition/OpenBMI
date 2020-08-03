detection= NaN*zeros(size(calib_set));

for setno= 1:3,
  file= [TODAY_DIR 'mmn_calib_set' int2str(set_no) VP_CODE];
  mrk_orig= eegfile_readBVmarkers(file);
  mrk= mrkodef_general_oddball(mrk_orig);
  iresp= find(~mrk.missingresponse);
  ishit= zeros(1, length(mrk.pos));
  ishit(iresp)= mrk.ishit;
  for ii= 1:4,
    idx= find(mrk.toe==20+ii);
    detection(setno, ii)= 100*mean(ishit(idx));
  end
end

figure;
set(gcf,'Pos',[20 200 640 480]);
[so,si]= sort(calib_set(:));
plot(calib_set(si), detection(si), '-o');
set(gca, 'YLim',[-4 104], 'XLim',calib_set(si([1 end]))+0.5*[-1;1]);
set(gca, 'XTick',calib_set(:));
xlabel('stimulus');
ylabel('detection rate  [%]');

selected_set= [];
while length(selected_set)~=3 | length(intersect(selected_set, calib_set(:)))~=3,
  fprintf('define variable <selected_set> according to tuning curve and type ''dbcont''.\n');
  keyboard
end
close(gcf)
