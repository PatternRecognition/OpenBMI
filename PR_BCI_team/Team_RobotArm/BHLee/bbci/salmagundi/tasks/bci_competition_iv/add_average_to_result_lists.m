res_dir= [DATA_DIR 'bci_competition_iv/evaluation/'];

name_list= {'1','1a','2a','2b','3','4'};
score_fmt= '%.2f';

for ds= 3:5,
  switch(name_list{ds});
   case '2a',
    nScores= 9;
   case '2b',
    nScores= 9;
   case '3',
    nScores= 2;
  end
  filein= [res_dir 'copy/evaluation_ds' name_list{ds} '.txt'];
  score= cell(1, nScores);
  [score{:}, name, lab, method]= ...
    textread(filein, ...
             [repmat('%f',[1 nScores]) '%s%s%s\n'], 'delimiter','|');
  score= [score{:}];
  avg_score= mean(score, 2);
  
  fileout= [res_dir 'evaluation_ds' name_list{ds} '.txt'];
  fid= fopen(fileout, 'wt');
  for ii= 1:length(name),
    fprintf(fid, [repmat([score_fmt ' | '],[1 nScores+1]) '%s | %s | %s\n'], ...
            avg_score(ii), score(ii,:), name{ii}, lab{ii}, method{ii});
  end
  fclose(fid);
  
end
