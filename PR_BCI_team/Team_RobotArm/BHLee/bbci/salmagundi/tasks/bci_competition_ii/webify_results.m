web_dir= '/home/nibbler/blanker/candy_Web/htdocs/bci/competition/results/';
sort_sgn= 1;
%file= 'tuebingen_ia_results'; score_fmt= '%2.1f%%'; score_name= 'error';
%file= 'tuebingen_ib_results'; score_fmt= '%2.1f%%'; score_name= 'error';
%file= 'albany_results'; score_fmt= '%2.1f%%'; score_name= 'error';
%file= 'graz_results'; score_fmt= '%4.2f'; score_name= 'MI/t'; sort_sgn=-1;
%file= 'berlin_results'; score_fmt= '%d%%'; score_name= 'error';

fmt_str= [' <td align="right", bgColor=#%s>' score_fmt '&nbsp;</td>\n'];
out_file= [web_dir file '.html'];


[name, score, lab, coname]= textread([web_dir file '.txt'], '%s%f%s%s\n', ...
                                     'delimiter','|');
if strcmp(score_name, 'error'),
  if all(score<1),
    score= 100*score;
  end
end
if length(coname)<length(name),
  coname= cat(1, coname, cell(length(name)-length(coname),1));
end
[so,si]= sort(sort_sgn*score);
name= name(si);
score= score(si);
lab= lab(si);
coname= coname(si);

  
fid= fopen(out_file, 'wt');

fprintf(fid, '<table>\n');
fprintf(fid, '<tr> <th>#.</th>\n');
fprintf(fid, ' <th>contributor</th> <th>&nbsp;%s&nbsp;</th>\n', score_name);
fprintf(fid, ' <th>research lab</th> <th>co-contributors</th> </tr>\n');
for ii= 1:length(name),
  if ii==1,
    colStr= 'FFC000';
  else
    colStr= 'DDDDDD';
  end
  rk= 1 + sum(score<score(ii));
  fprintf(fid, '<tr> <td align="right", bgColor=#%s>%d.</td>\n', ...
          colStr, rk);
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, deblank(name{ii}));  
  fprintf(fid, fmt_str, colStr, score(ii));
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, deblank(lab{ii}));
  fprintf(fid, ' <td bgColor=#%s>%s </td> </tr>\n', ...
          colStr, deblank(coname{ii}));
end
fprintf(fid, '</table>\n');
fprintf(fid, '<br/>\n\n\n');

for ii= 1:length(name),
  fprintf(fid, '<h4>%d. %s', ii, deblank(name{ii}));
  if ~isempty(lab{ii}),
    fprintf(fid, ', <i>%s</i>', deblank(lab{ii}));
  end
  fprintf(fid, '</h4>\n');
  if ~isempty(coname{ii}),
    fprintf(fid, 'with %s<br/>\n', deblank(coname{ii}));
  end
  fprintf(fid, '<i>Features:</i>\n');
  fprintf(fid, ' <br/>\n');
  fprintf(fid, '<i>Classification:</i>\n');
  fprintf(fid, ' <br/>\n');
  fprintf(fid, 'some details\n');
  fprintf(fid, ' [ <a href="_desc.txt">txt</a> ]\n');
  fprintf(fid, '<p/>\n\n\n');
end
fclose(fid);






file= 'albany_P300_results'; score_fmt= '%2.1f%%'; score_name= 'error';

fmt_str= [' <td align="right", bgColor=#%s>' score_fmt '&nbsp;</td>\n'];
out_file= [web_dir file '.html'];


[name, score, rep, lab, coname]= textread([web_dir file '.txt'], ...
                                          '%s%f%d%s%s\n', 'delimiter','|');

fid= fopen(out_file, 'wt');

fprintf(fid, '<table>\n');
fprintf(fid, '<tr> <th>#.</th>\n');
fprintf(fid, ' <th>contributor</th> <th>&nbsp;%s&nbsp;</th>\n', score_name);
fprintf(fid, ' <th>rep</th>\n');
fprintf(fid, ' <th>research lab</th> <th>co-contributors</th> </tr>\n');
for ii= 1:length(name),
  if ii==1,
    colStr= 'FFC000';
  else
    colStr= 'DDDDDD';
  end
  rk= 1 + sum( (score<score(ii)) | (score==score(ii) & rep<rep(ii)) );
  fprintf(fid, '<tr> <td align="right", bgColor=#%s>%d.</td>\n', ...
          colStr, rk);
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, deblank(name{ii}));  
  fprintf(fid, fmt_str, colStr, score(ii));
  fprintf(fid, ' <td bgColor=#%s>%d </td>\n', colStr, rep(ii));
  fprintf(fid, ' <td bgColor=#%s>%s </td>\n', colStr, deblank(lab{ii}));
  fprintf(fid, ' <td bgColor=#%s>%s </td> </tr>\n', ...
          colStr, deblank(coname{ii}));
end
fprintf(fid, '</table>\n');
fprintf(fid, '<br/>\n\n\n');

for ii= 1:length(name),
  fprintf(fid, '<h4>%d. %s', ii, deblank(name{ii}));
  if ~isempty(lab{ii}),
    fprintf(fid, ', <i>%s</i>', deblank(lab{ii}));
  end
  fprintf(fid, '</h4>\n');
  if ~isempty(coname{ii}),
    fprintf(fid, 'with %s<br/>\n', deblank(coname{ii}));
  end
  fprintf(fid, '<i>Features:</i>\n');
  fprintf(fid, ' <br/>\n');
  fprintf(fid, '<i>Classification:</i>\n');
  fprintf(fid, ' <br/>\n');
  fprintf(fid, 'some details\n');
  fprintf(fid, ' [ <a href="_desc.txt">txt</a> ]\n');
  fprintf(fid, '<p/>\n\n\n');
end
fclose(fid);
