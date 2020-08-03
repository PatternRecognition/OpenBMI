res_dir= [DATA_DIR 'bci_competition_iv/evaluation/'];
tex_dir= [TEX_DIR 'presentation/bci_competition_iv/results_poster/'];

name_list= {'1','1a','2a','2b','3','4'};

for ds= 1:length(name_list),

sort_sgn= -1;
lab_width= 6;
coname_width= 6;
score_fmt= '%.2f';
nShow= inf;
nScoresIn= 0;
switch(name_list{ds}),
 case '1',
  score_fmt= {'%.3f', '%.2f', '%.2f', '%.2f', '%.2f'};
  score_name= {'mse','a','b','f','g'};
  sort_sgn= 1;
  nShow= 7;
 case '1a',
  score_fmt= {'%.3f', '%.2f', '%.2f', '%.2f'};
  score_name= {'mse','c','d','e'};
  sort_sgn= 1;
  nShow= 6;
  coname_width= 7;
 case '2a',
  score_name= {'kappa'};
  nScoresIn= 10;
  lab_width= 9;
  coname_width= 9;
%  score_name= cat(2, {'kappa'}, cprintf('%d', 1:9)');
%  coname_width= 4;
 case '2b',
  score_name= {'kappa'};
  nScoresIn= 10;
  lab_width= 8;
  coname_width= 8;
%  score_name= cat(2, {'kappa'}, cprintf('%d', 1:9)');
%  coname_width= 5;
 case '3',
  score_fmt= '%.1f';
  score_name= {'acc', 'S1', 'S2'};
  lab_width= 8;
  coname_width= 7;
 case '4',
  score_name= {'r'};
  lab_width= 9;
 otherwise,
  error('unknown data set');
end
%nShow= 'all';

if isequal(nShow, 'all'),
  out_file= [tex_dir 'results_table_ds' name_list{ds} '.tex'];
  tablename= 'longtable';
else
  out_file= [tex_dir 'results_table_ds' name_list{ds} '_selected.tex'];
  tablename= 'tabular';
end


if ~iscell(score_name), score_name= {score_name}; end
nScores= length(score_name);
if nScoresIn==0,
  nScoresIn= nScores;
end
if ~iscell(score_fmt), score_fmt= repmat({score_fmt}, [1 nScores]); end
score= cell(1, nScoresIn);
[score{:}, name, lab, method]= ...
    textread([res_dir 'evaluation_ds' name_list{ds} '.txt'], ...
             [repmat('%f',[1 nScoresIn]) '%s%s%s\n'], 'delimiter','|');
score= [score{:}];
if nScores<nScoresIn
  score= score(:,1:nScores);
end
coname= cell(1, length(name));
for ii= 1:length(name),
  is= min(find(name{ii}==','));
  if ~isempty(is),
    coname{ii}= name{ii}(is+1:end);
    name{ii}= name{ii}(1:is-1);
  end
end
[so,si]= sort(sort_sgn*score(:,1));
name= unblank(name(si));
score= score(si,:);
lab= unblank(lab(si));
coname= unblank(coname(si));
method= unblank(method(si));


fid= fopen(out_file, 'wt');

fprintf(fid, '\\begin{small}\n');
fprintf(fid, '\\begin{%s}{@{}rl', tablename);
fprintf(fid, repmat('r', [1 nScores]));
fprintf(fid, ['p{' num2str(lab_width) 'cm}p{' num2str(coname_width) ...
              'cm}@{}}\n']);
fprintf(fid, ' \\textbf{\\#.}\n');
fprintf(fid, ' & \\textbf{contributor}\n');
fprintf(fid, ' & \\textbf{%s}\n', score_name{:});
fprintf(fid, ' & \\textbf{research lab}\n');
fprintf(fid, ' & \\textbf{co-contributors}\\\\\n');
if isequal(nShow, 'all'),
  N= length(name);
else
  N= min(nShow, length(name));
end
for ii= 1:N,
  if ii==1,
    ColStr= ' \\Violet{%s}';
    colStr= ' \\violet{%s}';
  else
    ColStr= ' \\textbf{%s}';
    colStr= ' %s';
  end
  rk(ii)= 1 + sum(sort_sgn*score(:,1)<sort_sgn*score(ii,1));
  fprintf(fid, colStr, [int2str(rk(ii)) '.']);
  fprintf(fid, [' &' colStr '\n'], name{ii});
  fprintf(fid, [' &' ColStr '\n'], sprintf(score_fmt{1}, score(ii,1)));
  for k= 2:nScores,
    fprintf(fid, [' &' colStr '\n'], ...
            sprintf(['{\\scriptsize ' score_fmt{k} '}'], score(ii,k)));
  end
  if isempty(lab{ii}),
    lab{ii}= ' ';
  end
  fprintf(fid, [' &' colStr '\n'], sprintf('{\\scriptsize %s}', lab{ii}));
  if isempty(coname{ii}),
    coname{ii}= ' ';
  end
  fprintf(fid, [' &' colStr], coname{ii});
  if ii<N,
    fprintf(fid, '\\\\');
  end
  fprintf(fid, '\n');
end
if N<length(name),
  fprintf(fid, '\\\\...\n');
end
fprintf(fid, '\\end{%s}\n', tablename);
fprintf(fid, '\\end{small}\n');

fclose(fid);

end
