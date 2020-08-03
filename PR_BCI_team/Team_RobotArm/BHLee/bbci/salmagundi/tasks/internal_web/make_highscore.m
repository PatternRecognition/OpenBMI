dd= dir([EEG_RAW_DIR '*_*']);

kk= 0;
clear score_list;
for ii= 1:length(dd),
  sub_dir= [EEG_RAW_DIR dd(ii).name '/log/'];
  if ~exist(sub_dir), continue; end
  ee= dir([sub_dir 'feedback_*.log']);
  scores= [];
  log_nr= [];
  for jj= 1:length(ee),
    s= extractScoresFromLog([sub_dir ee(jj).name], 100);
    if isempty(s),
      s= -inf;
    end
    scores= cat(2, scores, max(s));
  end
  if any(~isinf(scores)),
    kk= kk+1;
    is= min(find(dd(ii).name=='_'));
    [ma,mi]= max(scores);
    score_list(kk)= struct('subject',dd(ii).name(1:is-1), ...
                           'score', scores(mi), ...
                           'date', dd(ii).name(is+1:end), ...
                           'log_file', ee(mi).name(10:end-4));
  end
end
[so,si]= sort(-[score_list.score]);
score_list= score_list(si);

isBestOfSubject= ones(1,length(score_list));
for ii= 2:length(score_list),
  if ismember(score_list(ii).subject, {score_list(1:ii-1).subject}),
    isBestOfSubject(ii)= 0;
  end
end
iBestOfSubject= find(isBestOfSubject);
top_ten= score_list(iBestOfSubject(1:10));

  
fid= fopen([BCI_WEB_DIR 'experiments/high_scores.html'], 'w');
fprintf(fid, '<html>\n<head>\n<title>BBCI High Scores</title>\n</head>');
fprintf(fid, '\n<font face="sans-serif">\n');
fprintf(fid, '\n<body>\n');
fprintf(fid, '<h2>Top Ten BBCI Feedback Performers</h2>\n');
webify_feedbackScore(fid, top_ten);
fprintf(fid, '<p></p><br/>\n');
fprintf(fid, '<h2>High Scores obtained by BBCI Feedback</h2>\n');
webify_feedbackScore(fid, score_list);
fprintf(fid, '</body>\n</html>\n');
fclose(fid);

cmd= sprintf('pushd %s ; cvs commit -m "updated bbci high score list" high_scores.html ; popd', [BCI_WEB_DIR 'experiments']);
unix(cmd);
