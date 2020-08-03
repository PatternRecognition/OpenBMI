detection=zeros(2,2,2,4);

for set_no= 1:2;
  for word_no= 1:2;
    for speaker_no= 1:2;
      file= [TODAY_DIR 'bfnt_a3_season2_calib_set' int2str(set_no) int2str(word_no) int2str(speaker_no) VP_CODE];
      mrk_orig= eegfile_readBVmarkers(file);
      respDef= {'R128', 'right'};
      mrk= mrkodef_general_oddball(mrk_orig, 'respDef', respDef);
      iresp= find(~mrk.missingresponse);
      ishit= zeros(1, length(mrk.pos));
      ishit(iresp)= mrk.ishit;
      for ii= 1:4,
        idx= find(mrk.toe==20+ii);
        detection(set_no,word_no,speaker_no,ii)= 100*mean(ishit(idx));
      end
    end
  end
end

for set_no= 1:2;
  for ii= 1:4,
    detection_help(1,1)=detection(set_no,1,1,ii);
    detection_help(1,2)=detection(set_no,1,2,ii);
    detection_help(1,3)=detection(set_no,2,1,ii);
    detection_help(1,4)=detection(set_no,2,2,ii);
    detection_erg(set_no,ii)=mean(detection_help);
  end
end

figure;
set(gcf,'Pos',[20 200 640 480]);
[so,si]= sort(calib_set(:));
plot(calib_set(si), detection_erg(si), '-o');
set(gca, 'YLim',[-4 104], 'XLim',calib_set(si([1 end]))+0.5*[-1;1]);
set(gca, 'XTick',calib_set(:));
xlabel('bitrate');
ylabel('detection rate  [%]');

selected_set= [];
%while length(selected_set)~=4 | length(intersect(selected_set, calib_set(:)))~=3,
  fprintf('define variable <selected_set=[T1 T2 T3 T4]> according to tuning curve.\n');
%  keyboard
%end
%close(gcf)
