function info = write_log(data,varargin);
%WRITE_LOG writes log_files for feedbacks. Can be read with load_log 
% 
% usage : 
%   <init> :   write_log('init',name,fb_opt);
%   <exit> :   write_log('exit');
%   <flush> :  write_log('flush');
%   <data> :   write_log(propertylist);
%   <marker> : write_log(marker);
%   <info>:    info = write_log('???');
%
% input:
%   name      the name of the feedback, e.g. brainpong, 2d, basket
%   fb_opt    the initialization feedback opt struct
%   propertylist:  variable1, value1,.... (is written into the log file)
%       
%   The values will be written in the form:
%   variable1 = value1 § variable2 = value2 §....
%
%   marker    a marker number
%   info:     writing state??
%
%   The marker will be written in the form: MARKER = ...
% NOTE: Do not use = and § anywhere!!!!
%
% write_log will write log-files to LOG_DIR (global and not exmpty) 
% or to g:\eeg_temp\log\ (WINDOWS)  or \tmp\ (LINUX) with the following name:
% feedback_<name>_<number>.log 
% where name is the name given above and number is a ongoing free 
% number in the directory. If feedback_opt.parPort exist and is true, 
% ppTrigger(100+number) is sent,too.
% Additionally a feedback_<name>_<time>.mat is written to save the 
% feedback_opt environment, the name of this variable is also saved 
% in the log-file

persistent fid name stri newer lpoi status;

if isempty(status), status = 0;end
global LOG_DIR TODAY_DIR
if isempty(LOG_DIR)
  if isunix
    LOG_DIR = '/tmp/';
  else
    LOG_DIR = [TODAY_DIR '\log\'];
  end
end

if ischar(data) & strcmp(data,'init')
  status = 1;
  ve = datevec(now);
  v = cat(1, {num2str(ve(1))}, ...
            cellstr(char(max('0',double(num2str(round(ve(2:6))'))))));
  poi = 1;
  name = varargin{1};
  while exist(sprintf('%sfeedback_%s_%03i.log', ...
                        LOG_DIR,name,poi),'file'),
    poi = poi+1;
  end
  old_stri = stri;
  
  str = sprintf('%sfeedback_%s_%03i.log',LOG_DIR,name,poi);
  if ~isempty(fid) & fid>0 ; try;fclose(fid);end;end
  fid = fopen(str,'w');
  stri = sprintf('Feedback started at ',stri);
  stri = [stri,sprintf('%s_',v{1:end-1})];
  stri = sprintf('%s%s with values:\nfeedback_%s_fb_opt',stri,v{end},name);
  %stri = [stri,sprintf('_%s',v{:})];
  %stri = sprintf('%s\n\n',stri);
  stri = [stri,'_', sprintf('%03i',poi),char(10)];
  fb_opt = varargin{2};
  save([LOG_DIR 'feedback_' name '_fb_opt' ...
          sprintf('_%03i', poi)],'fb_opt');
  fprintf('Write log-file: %i\n',poi);
  if fb_opt.parPort
    ppTrigger(101+mod(poi-1,99));
  end
  if isfield(fb_opt,'init_file') & ~isempty(fb_opt.init_file)
    func = fb_opt.init_file;
  else
    func = [fb_opt.type '_init'];
  end
  ww = which(func);
  if isempty(ww)
    warning(sprintf('%s not found\n',func));
  else
    lstr = dir(ww);
    lstr = lstr.date;
    if isempty(newer) | ~strcmp(newer,lstr);
      newer = lstr;
      lpoi = 1;
      while exist(sprintf('%s/feedback_%s_init_%03i.m', ...
                          LOG_DIR,name,lpoi),'file'),
        lpoi = lpoi+1;
      end
      st= copyfile(ww, sprintf('%s/feedback_%s_init_%03i.m', LOG_DIR,name,lpoi));
      if st~=1
        newer = [];
      end
    end
    if ~isempty(newer)
      stri = sprintf('%sInit file copied to\nfeedback_%s_init_%03i.m\n\n',stri,name,lpoi);
    end
  end
  if ~isempty(old_stri)
     stri = sprintf('%s%s',stri,old_stri);
  end
  
  %   if poi>99
%     error('maximum number of log files exceeded: do garbage collection');
%   end
%   if poi>90
%     warning('more than 90 log files used: do garbage collection');
%   end
  return
end

if ischar(data) & strcmp(data,'???')
  info = status;
  return;
end

if ischar(data) & strcmp(data,'exit')
  try;fclose(fid);end
  stri = '';
  status = 0;
  fid = [];
  return
end

if ischar(data) & strcmp(data,'flush');
  if ~isempty(stri)
    try;fprintf(fid,'%s',stri);end
  end
  stri = '';
  return;
end

if ischar(data) & strcmp(data,'comment')
  stri = sprintf('%s § Comment: %s\n',varargin{1});
  return
end


if ischar(data),
  C= cat(2, {data}, varargin);
  for i = 1:2:length(C),
    if ischar(C{i+1}),
      valstr= sprintf('''%s''', C{i+1});
    elseif iscell(C{i+1}),
      valstr= sprintf('{%s}', sprintf('''%s'',', C{i+1}{:}));
      valstr(end-1)= [];  %% delete last ','
    elseif length(C{i+1})==1,
      if C{i+1}==round(C{i+1})
        valstr = sprintf('%1.0f',C{i+1});
      else
        valstr= sprintf('%f', C{i+1});
      end
    else
      valstr= sprintf('%f  ', C{i+1});
      valstr= ['[' valstr ']'];
    end
    stri= sprintf('%s%s = %s', stri, C{i}, valstr);
    if i<length(C)-1,
      stri= [stri ' § '];
    end
  end
  stri = sprintf('%s\n',stri);
  return
end

if isnumeric(data)
  stri = sprintf('%sMARKER = %d § counter = %d\n', stri, data, varargin{1});
end
