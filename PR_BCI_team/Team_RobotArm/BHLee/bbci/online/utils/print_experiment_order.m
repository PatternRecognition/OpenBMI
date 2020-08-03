function print_experiment_order(varargin)
% generates a random order of classifier applications for
% the zero-training-Experiments.

global EEG_RAW_DIR
if nargin==0
  % no arguments given:
  sub = input('Name of the subject:\n','s');
else
  sub = varargin{1};
end

today_str= datestr(now, 'yy/mm/dd');
today_str(find(today_str=='/'))= '_';
sub_dir= [sub '_' today_str];
file= [EEG_RAW_DIR sub_dir '/experiment_order.txt'];

if ~exist(file,'file')
  fid = fopen(file,'wt');
  if fid==-1,
    error(sprintf('cannot open file <%s> for writing', file));
  end
  
  fprintf(fid,'Experimentprotokoll\n\n\t Proband: %s;\n',sub);
  fprintf(fid,'\t Datum: %s\n\n',today_str);
  fprintf(fid,'0. Eingelen.\n');
  fprintf(fid,'1. Impedanzen.\n');
  fprintf(fid,'2. Artefaktmessung.\n');
  fprintf(fid,'3. CONCAT trainieren.\n');
  fprintf(fid,'4. 4 runs CONCAT Hau-den-Lukas a 10 min.\n');
  fprintf(fid,'5. CSP trainieren auf den Daten von Punkt 4.\n\n');
  
  coin = round(rand(1,4));
  
  for ii = 1:2
    for jj = 1:2
      num = (jj + (ii-1)*2);
      if jj ==1
	str = 'Cursor on.';
      else
	str = 'Cursor off.';
      end
      if coin(num)
	fprintf(fid,'%i. %s\n\tCONCAT Hau-den-Lukas a 10 min.\n',5+num,str);
	fprintf(fid,'\tCSP Hau-den-Lukas a 10 min.\n');
      else
	fprintf(fid,'%i. %s\n\tCSP Hau-den-Lukas a 10 min.\n', 5+num,str);
	fprintf(fid,'\tCONCAT Hau-den-Lukas a 10 min.\n');
      end
    end
  end
  fprintf(fid,'10. Impedanzen.');
  fclose(fid);
end

% print out the experiment list on hplj2:
eval(['! lpr -Phplj2 ' file]);