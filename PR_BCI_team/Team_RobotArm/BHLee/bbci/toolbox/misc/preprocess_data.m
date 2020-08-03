function preprocess_data(subject,varargin);
%PREPROCESS_DATA DOES SOME EASY PROCESSING OF DATA (FROM BRAINVISION FORMAT TO BRAINVISION FORMAT)
%
% usage:
%    preprocess_data(subject,opt);
%    preprocess_data({subject1,subject2},opt);
%
% input:
%    subject: name of the player
%    subject1,subject2 both player names
%    opt: struct with fields
%        .data_dir: directory where original data were saved (i:\eeg_temp)
%        .date: the date of the experiment in format yy_mm_dd (default: if in data_dir all .vmrk files have the same creation date this date is used, otherwise today with a warning)
%        .files: Cell array of files to process. A cell can be a cell again to note files to append.
%            File names may contain the wildcard '*' at the end.
%            Note: 'impedances' should not be the first in the list.
%        .process: cell array regarding files  defining the processing steps. Can be zero for interactive mode. If it is empty (default) some intelligent processing regarding file name will be done.
%        .result_dir: directory to place the subject directories, default: i:\EEG_Daten\
%        .machine: place to find the log_files, for one player the default is brainamp, for two players:{{'verdandi','brainamp'},{'urd','brainamp'}}
%        .copy_logfiles: flag. Default true: copy logfiles
%        .do_scp: flag. Default true: try to copy logfiles from linux by scp
%        .linux_machine: machine to get the logfiles, default:bot00
%        .save_name: cell array regarding files with the save_names. You can use $n for subject name. With 0 interactive mode on. Default: use file names
%        .extract_subject: subjects to extract (0 for interactive mode)
%        .username:  username for scp connection (default: empty, interactive mode)
%        .parPort: default: {'Stimulus','Response'}. Type of parport markers for each player
%        .used_name: used subject name, default player 1
%        .temp_dir: temporary directory for the copy_script.bat. default:
%                   d:\temp\
%        .putty_path: path to pscp.exe; only required if pscp.exe is not in
%                     the path.
%       
%
% Guido Dornhege, 28/03/06

if isunix
	filesep = '/';
else
	filesep = '\';
end


if ischar(subject)
  subject = {subject};
end

opt= propertylist2struct(varargin{:});
opt.subject = subject;


opt = set_defaults(opt,'data_dir','d:\data\eeg_temp\',...
    'putty_path','',...
    'temp_dir','d:\temp\',...
  'date',[],...
  'files',{},...
  'process',{},...
  'result_dir','d:\EEG_Daten\',...
  'machine',{},...
  'copy_logfiles',1,...
  'do_scp',1,...
  'linux_machine','bot00',...
  'save_name',{},...
  'username','',...
  'extract_subject',{},...
  'parPort',{'Stimulus','Response'},...
  'used_name',opt.subject{1});

if isempty(opt.machine)
  if length(opt.subject)==1
    opt.machine = {{'verdandi'},{'brainamp'}};
  else
    opt.machine = {{'verdandi','brainamp'},{'urd','brainamp'}};
  end
else    
  for ii = 1:length(opt.machine)
    if ~iscell(opt.machine{ii})
       opt.machine{ii} = {opt.machine{ii}};
    end
  end
end
opt.singlemachine = true;
for ii = 1:length(opt.machine{1})-1
    % This only checks for the first machine. 
    if ~strcmp(opt.machine{1}{ii},opt.machine{1}{ii+1})
        opt.singlemachine =false;
    end 
end 
    
 


d = dir([opt.data_dir,'*.vmrk']);

if isempty(opt.date)
  dd = [];
  for i = 1:length(d)
    gg = datestr(d(i).date,25);
    if i==1
      dd = gg;
    else
      if ~strcmp(gg,dd)
        dd = [];
        break;
      end
    end
  end
  
  if ~isempty(dd)
    dd(dd=='/')= '_';
  else
    warning('no dates or different dates are given!!! Choose today.');
    da = datevec(now);
    dd = sprintf('%02d_%02d_%02d',mod(da(1:3),100));
  end
  opt.date = dd;
end

d = {d(:).name};
fprintf('The following file were found:\n');
for i = 1:length(d)
  d{i} = d{i}(1:end-5);
  fprintf('%s\n',d{i});
end

fprintf('\n');

nFiles= length(opt.files);
missing= {};
existing= {};
for i = 1:nFiles,
  idx= strpatternmatch(opt.files{i}, d);
  if iscell(opt.files{i}),
    for j = 1:length(opt.files{i}),
      if isempty(strpatternmatch(opt.files{i}{j}, d)),
        missing= cat(2, missing, opt.files{i}(j));
      end
    end
    existing= cat(2, existing, {d(idx)});
  else
    if isempty(idx),
      missing= cat(2, missing, opt.files(i));
    else
      existing= cat(2, existing, d(idx));
    end
  end
end
if ~isempty(missing),
  error(sprintf('The following files were not found: %s.\n', vec2str(missing)));
end
opt.files= existing;

if isempty(opt.files)
  opt.files = d;
end

if isempty(opt.process)
  opt.process = cell(1,length(opt.files));
end

if isempty(opt.extract_subject)
  opt.extract_subject = cell(1,length(opt.files));
end

if isempty(opt.save_name)
  opt.save_name = cell(1,length(opt.files));
end

if ~iscell(opt.process)
  opt.process = {opt.process};
end

if length(opt.process)==1
  opt.process = repmat(opt.process,[1,length(opt.files)]);
end

if ~iscell(opt.extract_subject)
  opt.extract_subject = {opt.extract_subject};
end

if length(opt.extract_subject)==1
  opt.extract_subject = repmat(opt.extract_subject,[1,length(opt.files)]);
end

if ~iscell(opt.save_name)
  opt.save_name = {opt.save_name};
end

if length(opt.save_name)==1
  opt.save_name = repmat(opt.save_name,[1,length(opt.files)]);
end

% check for attachment

dd = opt.files;
numb = zeros(1,length(dd));
for i = 1:length(dd)
  if iscell(dd{i})
    dd{i} = '';
  else
    if length(dd{i})>=length(opt.used_name) & (all(dd{i}(end-length(opt.used_name)+1:end)==opt.used_name) | all(dd{i}(end-length(opt.used_name):end-1)==opt.used_name))
      %dd{i} = dd{i}(1:end-length(opt.used_name));
      if dd{i}(end)>'0' & dd{i}(end)<='9'
        numb(i) = str2num(dd{i}(end));
        dd{i} = dd{i}(1:end-1);
      end
    end
  end
end

[dum,dum,ind] = unique(dd);

aind = {};
for i = 1:max(ind)
  cc = find(ind==i);
  if ~isempty(dd{cc(1)}) & length(cc)>1
    [dum,ccc] = sort(numb(cc));
    cc = cc(ccc);
    fprintf('Do you want to append the following files:\n\n');
    fprintf('%s\n',opt.files{cc});
    R = '';
    while ~strcmp(R,'y') & ~strcmp(R,'n')
      R= input('Please answer with y/n: ','s');
      fprintf('\n');
    end
    fprintf('\n');
    if R=='y'
      aind = {aind{:},cc};
    end
    
  end
end

files = opt.files(setdiff(1:length(opt.files),[aind{:}]));
process = opt.process(setdiff(1:length(opt.files),[aind{:}]));
save_name = opt.save_name(setdiff(1:length(opt.files),[aind{:}]));
extract_subject = opt.extract_subject(setdiff(1:length(opt.files),[aind{:}]));
for i = 1:length(aind)
  files = {files{:},opt.files(aind{i})};
  process = {process{:},opt.process{aind{i}(1)}};
  save_name = {save_name{:},opt.save_name{aind{i}(1)}};
  extract_subject = {extract_subject{:},opt.extract_subject{aind{i}(1)}};
end

opt.files = files;
opt.process = process;
opt.save_name = save_name;
opt.extract_subject = extract_subject;


for i = 1:length(opt.files);
  file = opt.files{i};
  process = opt.process{i};
  save_name = opt.save_name{i};
  extract_subject = opt.extract_subject{i};
  if isempty(process)
    process = {};
  end
  if ~iscell(file)
    file = {file};
  end
  if isnumeric(save_name) & save_name==0
    save_name = input(['Please specify save name for ' file{1} ':'],'s');
  end
  if isempty(save_name)
    save_name = file{1};
    if length(opt.subject)>1
      c = strfind(save_name,opt.used_name);
      if ~isempty(c)
        if length(c)>1
          error('subject name appear more than once');
        end
        save_name = [save_name(1:c-1),'$n',save_name(c+length(opt.used_name):end)];
      end
    end  
  end
    
  if isnumeric(extract_subject) & extract_subject==0 & length(opt.subject)>1
    extract_subject = input(['please announce player numbers to extract for ' file{1} ': '],'s');
    if isempty(extract_subject) 
      extract_subject = 1:length(subject);
    else
      extract_subject = str2num(extract_subject);
    end
      
  end
  if isempty(extract_subject)
    extract_subject = 1:length(subject);
  end

  if isnumeric(process) & process==0
    suggest = {};
    if ~isempty(strmatch('impedances',file{1}));
      if length(opt.subject)>1
        suggest = 'split_impedances';
      else
        suggest = 'copy';
      end
    else
      if ~isempty(strmatch('arte',file{1}))
        suggest = {'artefact_transfer'};
      end
      if ~isempty(strfind(file{1},'fb')) & length(opt.subject)>1
        suggest = {'map_marker'};
      end
      suggest = {suggest{:},'make_bipolar'};
    end
    
    str = sprintf(' %s',suggest{:});
    process = {};
    sss = input(['Specify ' numstr(length(process)+1) '. processing for ' file{1} ' [' str ']: '],'s');
    while ~isempty(sss)
      process = {process{:},sss};
      sss = input(['Specify ' numstr(length(process)+1) '. processing for ' file{1} ': '],'s');
    end
    if isempty(process)
      process = suggest;
    end
    if length(process)==1
      process = process{1};
    end
  end

  if isempty(process)
    if ~isempty(strmatch('impedances',file{1}));
      if length(opt.subject)>1
        process = 'split_impedances';
      else
        process = 'copy';
      end
    else
      if ~isempty(strmatch('arte',file{1}))
        process = {'artefact_transfer'};
      end
      if ~isempty(strfind(file{1},'fb')) & length(opt.subject)>1
        process = {'map_marker'};
      end
      process = {process{:},'make_bipolar'};
    end
    
  end
  fprintf('\n');
  opt.process{i} = process;
  opt.extract_subject{i} = extract_subject;
  opt.save_name{i} = save_name;
end

fprintf('\n');

for i = 1:length(opt.files);
  pause(0.5);
  file = opt.files{i};
  process = opt.process{i};
  save_name = opt.save_name{i};
  extract_subject = opt.extract_subject{i};
  if isempty(process)
    process = {};
  end
  if ~iscell(file)
    file = {file};
  end
     
  fprintf('Process');
  fprintf(' %s',file{:});
  fprintf(':\n');
  
      
  if ischar(process) 
    if strcmp(process,'copy')
      if isunix
        order = 'cp'; 
      else
        order = 'copy';
      end
      for su = extract_subject
        for iiii = 1:length(file)
          for s = {'eeg','vhdr','vmrk'};
            str = sprintf('%s %s%s.%s %s%s_%s%s%s.%s',order,opt.data_dir,file{iiii},s{1},opt.result_dir,opt.subject{su},opt.date,filesep,save_name,s{1});
            fprintf('\n%s',str);
            system(str);
          end
        end
      end
      
    elseif strcmp(process,'split_impedances');
      for su = extract_subject
        for iiii = 1:length(file)
          split_impedances([opt.data_dir,file{iiii}],su,[opt.result_dir,opt.subject{su},'_',opt.date,'\',save_name]);
        end
      end
          
    else
      process = {process};
    end
  end
  
  if iscell(process)
    
    for ii = extract_subject
      if ii ==1
        clab = {'not','x*'};
      else
        clab = {'x*'};
      end
      clear cnt mrk hdr
      fprintf('Load the data for %s\n',opt.subject{ii});
      for iii = 1:length(file)
        outputStructArray = ~isempty(strcmp(file{iii},'arte'));
        [cn,mr]= eegfile_loadBV([opt.data_dir file{iii}],'fs','raw','prec',1,'clab',clab,'channelwise',1,'outputStructArray',outputStructArray);
        if iii==1
          cnt = cn;
          mrk = mr;
        else
          [cnt,mrk] = proc_appendCnt(cnt,cn,mrk,mr);
        end
        clear cn mr
      end
      if ii==2 
        for iii = 1:length(cnt.clab)
          cnt.clab{iii} = cnt.clab{iii}(2:end);
        end
      end
      
      for j = 1:length(process)
        fprintf('\n%s',process{j});
        [cnt,mrk] = feval(process{j},cnt,mrk,ii,opt);
      end
      
      file2 = save_name;
      ccc = strfind(file2,'$n');
      while ~isempty(ccc)
        file2 = [file2(1:ccc(1)-1), opt.subject{ii},file2(ccc(1)+2:end)];
        ccc = strfind(file2,'$n');
      end
        
      cnt.title = [opt.result_dir,opt.subject{ii},'_',opt.date,filesep,file2];
      fprintf('\nSave in progress: %s',cnt.title);
      create_directory([opt.result_dir,opt.subject{ii},'_',opt.date,filesep]);
      eegfile_writeBV(cnt,mrk);
      fprintf('\n\n');
    end
  end
  
  fprintf('\n\n');
  
  
end

% copying log file

instructions = {};
if opt.copy_logfiles
  if isunix
    warning('can not copy log file in linux')
  else
    if opt.singlemachine
        %TODO
        for i = 1:length(opt.subject)
            create_directory([opt.result_dir,opt.subject{i},'_',opt.date,filesep,'log',filesep]);
            str = sprintf('copy %slog\* %s', [], [opt.result_dir,opt.subject{i},'_',opt.date,filesep,'log',filesep]);
            system(str);
        end
    else
        for i = 1:length(opt.subject)
            create_directory([opt.result_dir,opt.subject{i},'_',opt.date,filesep,'log',filesep]);
            for j = length(opt.machine{i}):-1:1
                str = ['copy \\' opt.machine{i}{j} '\data\log\* '  opt.result_dir,opt.subject{i},'_',opt.date,filesep,'log',filesep];
                system(str);
            end
        end
    end
  end
  
  % copy linux files
  if opt.do_scp
    if isempty(opt.username)
      opt.username = input('To get the linux log files please type in your user name: ','s');
    end
    
    fprintf('Preparing data transmission from linux... Password required several times (in different command window):\n\n');
    
    fid = fopen([opt.temp_dir 'dummes_copy_script.bat'],'w');
    for i = 1:length(opt.subject)
      str = [opt.username '@' opt.linux_machine ':/home/neuro/data/BCI/bbciRaw/' opt.subject{i} '_' opt.date '/' '*.mat'];
      str = [opt.putty_path 'pscp.exe -r ' str ' ' opt.result_dir,opt.subject{i},'_',opt.date,filesep];
      fprintf('%s\n',str);
      fprintf(fid,'%s\n',str);
      str = [opt.username '@' opt.linux_machine ':/home/neuro/data/BCI/bbciRaw/' opt.subject{i} '_' opt.date '/' '*_log'];
      str = [opt.putty_path 'pscp.exe -r ' str ' ' opt.result_dir,opt.subject{i},'_',opt.date,filesep];
      fprintf('%s\n',str);
      fprintf(fid,'%s\n',str);
    end
    fprintf(fid,'pause\nexit\nexit\n');
    fclose(fid);
    system([ opt.temp_dir 'dummes_copy_script.bat &']);
    instructions = {instructions{:},'Please use the cmd window to transfer the files (password required)'};
  else
    if opt.singlemachine
        for ii = 1:length(subject)
            error('files = %TODO: Recursive copying. Files sind in D:/data/eeg_temp/');
            str = sprintf('copy %s %s',[],[opt.result_dir opt.subject{ii} '_' opt.date filesep]); 
            fprintf('%s\n',str);
            system(str);
        end 
    else
       str = 'Get the log files on processing side: \n'
       for i = 1:length(opt.subject)
         str = [str, '/home/neuro/data/BCI/bbciRaw/' opt.subject{i} '_' opt.date '/' '*.mat ' opt.result_dir,opt.subject{i},'_',opt.date,filesep,'\n'];
         str = [str, '/home/neuro/data/BCI/bbciRaw/' opt.subject{i} '_' opt.date '/' '*_log ' opt.result_dir,opt.subject{i},'_',opt.date,filesep,'\n'];
       end
       instructions = {instructions{:},str};
    end
  end
end

if length(opt.subject)==2
  instructions = {instructions{:},sprintf('Burn the directories (EEG_Daten) %s_%s and %s_%s (and maybe eeg_temp)',opt.subject{1},opt.date, opt.subject{2},opt.date)};
  instructions = {instructions{:},sprintf('Copy the directories (EEG_Daten) %s_%s and %s_%s to linux',opt.subject{1},opt.date, opt.subject{2},opt.date)};
else
  instructions = {instructions{:},sprintf('Burn the directory %s_%s',opt.subject{1},opt.date)};
  instructions = {instructions{:},sprintf('Copy the directory %s_%s to linux',opt.subject{1},opt.date)};  
end
  
instructions = {instructions{:},'Clean up all log directories and eeg_temp directories','prepare the data (prepare_data_bbci_bet) on linux'};

fprintf('Data prepared!!!\n\n')

if length(instructions)>0
  fprintf('Please do the following things:\n');
  for i = 1:length(instructions)
    fprintf('%i. %s\n',i,instructions{i});
  end
  fprintf('\n\n');
end

return


function [cnt,mrk] = artefact_transfer(cnt,mrk,player,opt);
% For some reason, this mrk appears not to be a struct array.
if length(mrk)==1
  mrk= eegfile_readBVmarkers(cnt.file, 1);
  for ii = 1:length(mrk)
    mrk(ii).pos= round( mrk(ii).pos/mrk(ii).fs*cnt.fs );
    mrk(ii).fs= cnt.fs; 
  end
end
for i = 1:length(mrk)
  if strcmp(mrk(i).type,'Stimulus') & str2num(mrk(i).desc(2:end))<=13
    mrk(i).type = 'Comment';
    mrk(i).desc = getfield({'Augen links','Augen rechts','Augen oben', 'Augen unten', 'blinzeln', 'Augen zu & entspannen', 'Augen offen & entspannen', 'beißen', 'Kopf bewegen', 'stopp','EMG links','EMG rechts','EMG fuss'},{str2num(mrk(i).desc(2:end))});
    mrk(i).desc = mrk(i).desc{1};
  end
end


return;

function [cnt,mrk] = map_marker(cnt,mrk,player,opt);

typ = opt.parPort{player};
ntyp = opt.parPort{3-player};

ind = [];
for i = 1:length(mrk)
  if strcmp(mrk(i).type,ntyp);
    ind = [ind,i];
  else
    if strcmp(mrk(i).type,typ)
      mrk(i).type = 'Stimulus';
      mrk(i).desc(1) = 'S';
    end
  end
end

mrk = mrk(setdiff(1:length(mrk),ind));

return;



function [cnt,mrk] = make_bipolar(cnt,mrk,player,opt);
%% Only first input and output argument is used.
%% The rest is defined for formal reasons.

bil = [];
for s = {'EOG','EMG'}
  ch = cnt.clab;
  ind= strmatch(s{1}, ch)';
  ch = ch(ind);
  dol = [];
  for i = 1:length(ch)
    if ismember(i,dol)
      continue;
    end
    b = ch{i}(1:length(s{1})+1);
    ind2 = chanind(ch, strcat(b, {'p','n'}));
    if length(ind2)==2
      bil = cat(1,bil,ind(ind2));
    else
      warning('cannot make %s bipolar', b);
    end
    dol = [dol,ind2];
  end
end

if ~isempty(bil)
  for i = 1:size(bil,1)
    cnt.x(:,bil(i,1)) = int16(double(cnt.x(:,bil(i,1)))-(cnt.scale(bil(i,2))/cnt.scale(bil(i,1)))*double(cnt.x(:,bil(i,2))));
    cnt.clab{bil(i,1)} = cnt.clab{bil(i,1)}(1:end-1);
  end
  try,
    cnt.x(:,bil(:,2)) = [];
    cnt.scale(:,bil(:,2)) = [];
    cnt.clab = cnt.clab(setdiff(1:length(cnt.clab),bil(:,2)));
  catch
    %% in case of memory problems, keep those channels
    cnt.x(:,bil(:,2)) = NaN;
    cnt.scale(:,bil(:,2)) = 0;
    cnt.clab(bil(:,2))= {'NaC'};
  end
end

return;

function create_directory(direc);

while direc(end)==filesep
  direc = direc(1:end-1);
end

if ~exist(direc)
  c = strfind(direc,filesep);
  create_directory(direc(1:c(end)));
  mkdir(direc(1:c(end)),direc(c(end)+1:end));
end

  

