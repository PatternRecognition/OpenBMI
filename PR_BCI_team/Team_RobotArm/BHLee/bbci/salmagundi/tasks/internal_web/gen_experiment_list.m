expbase= readDatabase;

td_tag= '<td bgColor=#DDDDDD>';
fid= fopen([BCI_WEB_DIR 'experiments/experiment_details.txt'], 'w');
fid2= fopen([BCI_WEB_DIR 'experiments/experiment_details.html'], 'w');
fprintf(fid2, '<html>\n<head>\n<title>Interne BBCI Seite</title>\n</head>');
fprintf(fid2, '\n\n<body>\n');
fprintf(fid2, '<h2>List of experiments conducted in the BBCI project</h2>\n');
fprintf(fid2, '<table>\n');
for ie= 1:length(expbase),
  ll= length(expbase(ie).subject);
  dateStr= expbase(ie).file(ll+2:ll+9);
  evStr= '';
  try
    [mrk,mrk,mnt]= loadProcessedEEG(expbase(ie).file, '', {'mrk','mnt'});
    nChans= length(mnt.clab);
  catch
    mrk= [];
  end
  if isempty(mrk) | ~isfield(mrk, 'className'),
    try
      [clab,d,d,d, len]= readGenericHeader(expbase(ie).file);
      nChans= length(clab);
      evStr= sprintf('length: %d seconds', round(len));
    end
  else
    nClasses= min(length(mrk.className), size(mrk.y,1));
    for ic= 1:nClasses,
      evStr= [evStr, sprintf('%s [%d]', mrk.className{ic}, sum(mrk.y(ic,:)))];
      if ic<nClasses,
        evStr= [evStr ', '];
      end
    end
  end
  expStr= [expbase(ie).paradigm, expbase(ie).appendix];
  spStr= repmat(' ', [1 max(1, 20-length(expStr))]);
  fprintf(fid, '%s %8s - %s%s: %s\n', dateStr, expbase(ie).subject, ...
               expStr, spStr, evStr);
  fprintf(fid2, '<tr> %s%s&nbsp;</td>\n %s&nbsp;%s&nbsp;</td>\n', ...
          td_tag, dateStr, td_tag, expbase(ie).subject);
  fprintf(fid2, ' %s&nbsp;%s&nbsp;</td>\n', td_tag, expStr);
  fprintf(fid2, ' %s align="right">&nbsp;%d&nbsp;</td>\n', ...
          td_tag(1:end-1), nChans);
  fprintf(fid2, ' %s%s </td> </tr>\n', td_tag, evStr);
end
fclose(fid);
fprintf(fid2, '</body>\n</html>\n');
fclose(fid2);

cmd= sprintf('cd %s ; cvs commit -m "updated eeg experiment list" experiment_details.*', [BCI_WEB_DIR 'experiments']);
unix(cmd);
