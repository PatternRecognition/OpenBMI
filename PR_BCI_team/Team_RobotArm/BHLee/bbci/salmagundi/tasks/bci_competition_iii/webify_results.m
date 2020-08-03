res_dir= [DATA_DIR 'results/bci_competition_iii_submissions/'];
web_dir= '/mnt/share/neuro_www/webpages/projects/bci/competition_iii/results/';

fmt_str= [' <td align="right", bgColor=#%%s>%s&nbsp;</td>\\n'];


for ds= 1:8,

sort_sgn= -1;
lab_width= 8;
coname_width= 8;
switch(ds),
 case 1,
  file= 'tuebingen'; score_fmt= '%d%%'; 
  score_name= 'acc';
 case 2,
  file= 'albany'; score_fmt= '%3.1f%%'; 
  score_name= {'acc (15)', 'acc (5)'};
 case 3,
  file= 'graz_IIIa'; score_fmt= '%6.4f'; 
  score_name= {'kappa', 'K3','K6','L1'}; 
 case 4,
  file= 'graz_IIIb'; score_fmt= '%6.4f'; 
  score_name= {'MI/t','O3','S4','X11'}; 
 case 5,
  file= 'berlin_IVa'; 
  score_fmt= {'%5.2f%%','%4.1f%%','%4.1f%%','%4.1f%%','%4.1f%%','%4.1f%%'}; 
  score_name= {'acc', 'aa','al','av','aw','ay'}; 
 case 6,
  file= 'berlin_IVc'; score_fmt= '%.2f'; 
  score_name= 'mse'; sort_sgn= 1;
 case 7,
  file= 'martigny'; score_fmt= '%.2f'; 
  score_name= {'acc','s1','s2','s3'};
  case 8,
   file= 'martigny0'; score_fmt= '%.2f'; 
   score_name= {'acc','s1','s2','s3'}; 
end


out_file= [web_dir file '.html'];

if ~iscell(score_name), score_name= {score_name}; end
nScores= length(score_name);
if ~iscell(score_fmt), score_fmt= repmat({score_fmt}, [1 nScores]); end
score= cell(1, nScores);
[score{:}, name, lab, method]= ...
    textread([res_dir file '_results.txt'], [repmat('%f',[1 nScores]) '%s%s%s\n'], 'delimiter','|');
score= [score{:}];
if strcmp(file, 'berlin_IVa'),
  score= 100-score;
end
if strcmp(file(1:4), 'mart'),
  score= score(:,[4 1 2 3]);
end
is= min(find(file=='_'));
if isempty(is),
  provider= file;
else
  provider= file(1:is-1);
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
  

if ds<7,
  
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


else   %%  ds >= 7
  
precomp= method;
method(:)= {''};

fid= fopen(out_file, 'wt');

fprintf(fid, '<table>\n');
fprintf(fid, '<tr> <th>#.</th>\n');
fprintf(fid, ' <th>contributor</th>');
fprintf(fid, ' <th>psd</th>');
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
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, name{ii});  
  fprintf(fid, ' <td align="center", bgColor=#%s>%s </td>\n', ...
          colStr, precomp{ii});  
  fprintf(fid, sprintf(fmt_str, ['<b>' score_fmt{1} '</b>']), ...
          colStr, score(ii,1));
  for k= 2:nScores,
    fprintf(fid, sprintf(fmt_str, score_fmt{k}), colStr, score(ii,k));
  end
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, lab{ii});
  fprintf(fid, ' <td bgColor=#%s>%s </td> </tr>\n', ...
          colStr, coname{ii});
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

end





fid= fopen([web_dir 'index0.html'], 'r');
fod= fopen([web_dir 'index.html'], 'wt');
while ~feof(fid),
  line= fgetl(fid);
  fprintf(fod, '%s\n', line);
  if strncmp(line, '<!-- input', 10),
    inputfile= line(12:end-4);
    fud= fopen([web_dir inputfile], 'r');
    str= fread(fud);
    fprintf(fod, '%s', char(str'));
    fclose(fud);
  end
end
fclose(fod);
fclose(fid);
