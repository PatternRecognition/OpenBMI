res_dir= [DATA_DIR 'bci_competition_iv/evaluation/'];
web_dir= '~/neuro_www/webpages/projects/bci/competition_iv/results/';

fmt_str= [' <td align="right", bgColor=#%%s>%s&nbsp;</td>\\n'];

name_list= {'1','1a','2a','2b','3','4'};

for ds= 1:length(name_list),

sort_sgn= -1;
lab_width= 8;
coname_width= 8;
score_fmt= '%.2f';
nShow= 'all';
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
  score_name= cat(2, {'kappa'}, cprintf('%d', 1:9)');
  coname_width= 4;
 case '2b',
  score_name= {'kappa'};
  score_name= cat(2, {'kappa'}, cprintf('%d', 1:9)');
  coname_width= 5;
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

file= ['ds' name_list{ds}];
out_file= [web_dir 'results_table_' file '.html'];

if ~iscell(score_name), score_name= {score_name}; end
nScores= length(score_name);
if ~iscell(score_fmt), score_fmt= repmat({score_fmt}, [1 nScores]); end
score= cell(1, nScores);
[score{:}, name, lab, method]= ...
    textread([res_dir 'evaluation_ds' name_list{ds} '.txt'], ...
             [repmat('%f',[1 nScores]) '%s%s%s\n'], 'delimiter','|');
score= [score{:}];
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

fprintf(fid, '<table>\n');
fprintf(fid, '<tr> <th>#.</th>\n');
fprintf(fid, ' <th>contributor</th>');
fprintf(fid, ' <th>&nbsp;%s&nbsp;</th>\n', score_name{:});
fprintf(fid, ' <th>research lab</th> <th>co-contributors</th> </tr>\n');
for ii= 1:length(name),
  if ii==1,
    colStr= 'FFC000';
  else
    colStr= 'DDDDDD';
  end
  rk(ii)= 1 + sum(sort_sgn*score(:,1)<sort_sgn*score(ii,1));
  fprintf(fid, '<tr> <td align="right", bgColor=#%s>%d.</td>\n', ...
          colStr, rk(ii));
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, deblank(name{ii}));  
  fprintf(fid, sprintf(fmt_str, ['<b>' score_fmt{1} '</b>']), colStr, score(ii,1));
  for k= 2:nScores,
    fprintf(fid, sprintf(fmt_str, score_fmt{k}), colStr, score(ii,k));
  end
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, deblank(lab{ii}));
  fprintf(fid, ' <td bgColor=#%s>%s </td> </tr>\n', ...
          colStr, deblank(coname{ii}));
end
fprintf(fid, '</table>\n');
fprintf(fid, '<br/>\n\n\n');

for ii= 1:length(name),
  fprintf(fid, '<h4>%d. %s', rk(ii), name{ii});
  if ~isempty(lab{ii}),
    fprintf(fid, ', <i>%s</i>', lab{ii});
  end
  fprintf(fid, '</h4>\n');
  if ~isempty(coname{ii}),
    fprintf(fid, 'with %s<br/>\n', coname{ii});
  end
  if ~isempty(method{ii}),
    fprintf(fid, '%s\n <br/>\n', method{ii});
  end
  del_chars= ' ''.';
  desc_name= name{ii}(find(~ismember(name{ii}, del_chars)));
  ddd= dir([web_dir file '/' desc_name '_desc.*']);
  if ~isempty(ddd),
    fprintf(fid, 'some details\n');
    [dmy, desc_file, desc_ext]= fileparts(ddd.name);
    desc_ext= strrep(desc_ext, '.', '');
    fprintf(fid, ' [ <a href="%s/%s_desc.%s">%s</a>', ...
            file, desc_name, desc_ext, desc_ext);
    if exist([web_dir file '/' desc_name '_desc_orig.doc']),
      fprintf(fid, ' | <a href="%s/%s_desc_orig.doc">doc</a>', file, desc_name);
    end
    fprintf(fid, ' ]\n');
  end
  fprintf(fid, '<p/>\n\n\n');
end
fclose(fid);

end





fid= fopen([web_dir 'index0.html'], 'r');
fod= fopen([web_dir 'index_preview.html'], 'wt');
%fod= fopen([web_dir 'index.html'], 'wt');
while ~feof(fid),
  line= fgetl(fid);
  fprintf(fod, '%s\n', line);
  if strncmp(line, '<!-- input', 10),
    inputfile= line(12:end-4);
    fud= fopen([web_dir inputfile], 'r');
    str= char(fread(fud)');
    str= strrep(str, '\''e', '&eacute;');
    fprintf(fod, '%s', str);
    fclose(fud);
  end
end
fclose(fod);
fclose(fid);
