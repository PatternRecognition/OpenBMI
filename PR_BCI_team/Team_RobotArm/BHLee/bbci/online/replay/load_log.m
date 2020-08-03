function [out,counter,init_file] = load_log(data,varargin);
% LOAD_LOG loads a log_file.
% 
% usage:
%  <init>    fb_opt = load_log(name,number);
%  <init>    fb_opt = load_log(name);
%  <data>    [out,counter] = load_log;
%  <exit>    load_log('exit');
%
% input:
%
%  name       the name of the feedback (or directly name_number)
%  number     the number of the feedback
%
% output
%  fb_opt     the loaded fb_opt
%  out        the propertylist as cell array of this line or the marker as numeric array
%  counter    counter is the counter number
% 
% Guido Dornhege, 04/03/2004

persistent fid point;

global show_progress

global LOG_DIR
if isempty(LOG_DIR)
  if isunix
    LOG_DIR = '/tmp/';
  else
    LOG_DIR = 'i:\eeg_temp\log\';
  end
end

if exist('data','var') & ischar(data) & ~strcmp(data,'exit')
  if length(varargin)>0
    data = sprintf('%sfeedback_%s_%03i.log',LOG_DIR,data,varargin{1});
    if ~exist(data,'file')
      data = sprintf('%sfeedback_%s_%02i.log',LOG_DIR,data,varargin{1});
    end
  else
    data = sprintf('%sfeedback_%s.log',LOG_DIR,data);
  end
  
  if ~exist(data,'file')
    error('file does not exist');
  end
  
  if ~isempty(fid) & fid>0 ; try;fclose(fid);end;end
  if nargout>2
    fid = fopen(data,'r');
    init_file = '';
    while ~feof(fid)
      s = fgets(fid);
      if ~isempty(strmatch(s,'Init file copied to'))
        init_file = fgets(fid);
        break;
      end
    end
    fclose(fid);
  end

  fid = fopen(data,'r');
  if ~isempty(show_progress) & show_progress
    fseek(fid,0,1);
    point = ftell(fid);
    fseek(fid,0,-1);
  end
  counter = point;
end


if exist('data','var') & ischar(data) & strcmp(data,'exit');
  try; fclose(fid); end
  fid = [];
  return
end
  

if isempty(fid) | fid==-1
  out = [];counter = [];
  return
end
  

s = '';

while isempty(s) & ~feof(fid)
  s = fgets(fid);
  if isnumeric(s) & s==-1
    s = '';
  else
    s = deblank(s);
  end
end


if (isempty(s) | s==-1) & feof(fid)
  out= []; counter= [];
  return;
end

if ~isempty(show_progress) & show_progress
  fprintf('\r %i/%i       ',ftell(fid),point);
end

if strncmp('Feedback started',s,15);
  s = fgets(fid);
  s = deblank(s);
  ss = min(find((s==' ')==0));
  s = s(ss:end);
  ts = s(end-2:end);
  ii = min(find(ts>=48 & ts<=57));
  ts = ts(ii:end);
  s(end-length(ts)+1:end-length(ts)+3) = sprintf('%03i',str2num(ts));
  out = load([LOG_DIR s]);
  out = out.fb_opt;
  return;
end

if strncmp('Feedback changed',s,15);
  s = fgets(fid);
  s = deblank(s);
  ss = min(find((s==' ')==0));
  s = s(ss:end);
  out = load([LOG_DIR s]);
  out = out.fb_opt;
  return;
end

if strncmp('Init file copied to',s,15);
  s = fgets(fid);
  return;
end


while isempty(s) & ~feof(fid)
  s = fgets(fid);
  if isnumeric(s) & s==-1
    s = '';
  else
    s = deblank(s);
  end
end


if (isempty(s) | s==-1) & feof(fid)
  out= []; counter= [];
  return;
end

c = [0,strfind(s,'§'),length(s)+1];
out = cell(1,2*length(c)-2);

for i = 1:length(c)-1
  str = s(c(i)+1:c(i+1)-1);
  str = deblank(str);
  st = min(find((str==' ')==0));
  str = str(st:end);
  d = strfind(str,'=');
  nam = str(1:d(1)-1);
  out{2*i-1} = deblank(nam);
  val = str(d(1)+1:end);
  st = min(find((val==' ')==0));
  val = val(st:end);
  out{2*i} = fromString(val);
end

counter = [];
if strcmp(out{1},'MARKER')
  counter = out{4};
  out = out{2};
end

if strncmp('BLOCKTIME',s,9)
  out = struct('time',out{2},'logfile',out{4});
end
  