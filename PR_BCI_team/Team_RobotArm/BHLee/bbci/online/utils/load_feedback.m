function log = load_feedback(filename,fs);

global EEG_RAW_DIR
if isunix 
  if filename(1)~='/'
    filename = [EEG_RAW_DIR filename];
  end
else
  if filename(2)~=':'
    filename = [EEG_RAW_DIR filename];
  end
end

if ~exist('fs','var') | isempty(fs)
  fs = 100;
end

lag = fs/1000;

if ~exist(filename,'file')
  % some logfile or log directory seems to be missing.
  error(['File ' filename ' does not exist.']);
end
if isdir(filename)
  log = struct([]);
  d = dir([filename '/*.log']);
  for i = 1:length(d);
    fprintf('\rLoading Logfile %s (%d/%d)  ',d(i).name,i,length(d));
    log_new= load_feedback([filename '/' d(i).name],fs);
    if ~isempty(log_new) & length(fieldnames(log_new))<4,
      warning(sprintf('corrupted log file %s: skipping.', d(i).name));
    else
      if isfield(log, 'init_file') & ~isfield(log_new, 'init_file')
      	log_new.init_file = [];
      end
      log = cat(1, log, log_new);
    end
  end
  return;
end

log = struct('mrk',struct('pos',[],'toe',[],'counter',[],'lognumber',[]));
log.update = struct('lognumber',[],'counter',[],'pos',[],'object',[],'prop',{{}},'prop_value',{{}});
log.fs = fs;

isnanblock = true;
isnanblockfirst = true;
time = NaN;
lognumber = 0;
updateprop_pos = 0;
fid = fopen(filename, 'r', 'native', 'latin-1');
while ~feof(fid);
  s = fgets(fid);
  if isequal(s, -1), log=[]; continue; end  %% log file empty
  if length(s)<=1
    continue;
  end
  if strncmp(s,'Feedback started', length('Feedback started')),
    s = s(length('Feedback started at ')+1:length('Feedback started at ')+19);
    log.start = [str2int(s(1:4)),str2int(s(6:7)),str2int(s(9:10)),str2int(s(12:13)),str2int(s(15:16)),str2int(s(18:19))];
    s = fgets(fid);
    log.file = s(1:end-1);
    continue
  end
  if strncmp(s,'Init file copied to', length('Init file copied to')),
    s = fgets(fid);
    cc = strfind(filename,filesep);
    if isempty(cc), cc = length(filename)+1;end
    log.init_file = [filename(1:cc(end)-1) filesep s(1:end-1)];
    s = fgets(fid);
    continue;
  end
  if strncmp(s, 'MARKER', length('MARKER')),
    c = strfind(s,char(167));
    log.mrk.toe = [log.mrk.toe,str2int(s(10:c(1)-2))];
    s = s(c(1)+1:end);
    c = strfind(s,'=');
    s = s(c(1)+2:end-1);
    log.mrk.counter = [log.mrk.counter,str2int(s)];
    log.mrk.pos= [log.mrk.pos,time];
    log.mrk.lognumber= [log.mrk.lognumber,lognumber];
    continue
  end
  if strncmp(s,'counter', length('counter')),
    updateprop_pos = updateprop_pos +1;
    if updateprop_pos>length(log.update.prop)
      log.update.counter = cat(2,log.update.counter,zeros(1,2000));
      log.update.object = cat(2,log.update.object,zeros(1,2000));
      log.update.pos = cat(2,log.update.pos,zeros(1,2000));
      log.update.lognumber = cat(2,log.update.lognumber,zeros(1,2000));
      log.update.prop = cat(2,log.update.prop,repmat({{}},[1 2000]));
      log.update.prop_value = cat(2,log.update.prop_value,repmat({{}},[1 2000]));
    end
    
    c = strfind(s,'=');d = strfind(s,char(167));
    if isempty(d)
    	warning('Logfile was probably not correctly completed, writing was interrupted in between')
   	continue
   	end
   	
    log.update.counter(updateprop_pos) = str2int(s(c(1)+2:d(1)-2));
    if isnanblock
      if isnanblockfirst
        warning('Missing BLOCKTIME. I try to reconstruct!');
        isnanblockfirst = false;
      end
      time = 40*log.update.counter(updateprop_pos);
    end
    
    s = s(d(1)+2:end);
    c = strfind(s,'=');d = strfind(s,char(167));
    if isempty(d)
    	warning('Logfile was probably not correctly completed, writing was interrupted in between')
   		continue
   	end
    log.update.object(updateprop_pos) = str2int(s(c(1)+2:d(1)-2));
    log.update.pos(updateprop_pos) = time;
    log.update.lognumber(updateprop_pos)= lognumber;
    s = s(d(1)+2:end);
    c = strfind(s,'=');d = strfind(s,char(167));
    
    while ~isempty(d)
      log.update.prop{updateprop_pos} = cat(2,log.update.prop{updateprop_pos},{s(1:c(1)-2)});
      bla = s(c(1)+2:d(1)-2);
      try
        bla = eval(bla);
      end
      log.update.prop_value{updateprop_pos} = cat(2,log.update.prop_value{updateprop_pos},{bla});
      s = s(d(1)+2:end);
      c = strfind(s,'=');d = strfind(s,char(167));
    end
    d = length(s);
    log.update.prop{updateprop_pos} = cat(2,log.update.prop{updateprop_pos},{s(1:c(1)-2)});
    log.update.prop_value{updateprop_pos} = cat(2,log.update.prop_value{updateprop_pos},{eval(s(c(1)+2:d(1)-1))});
    
    continue
  end
  if strncmp(s, 'BLOCKTIME', length('BLOCKTIME')),
    d = strfind(s,char(167));  % char(167) = '§'
    isnanblock = false;
    if ~isempty(d);
      st = s(d(1)+1:end);
      c = strfind(st,'=');
      st =  st(c(1)+2:end-1);
      lognumber = str2int(st);
      s = s(1:d(1)-1);
    end
    isep = [find(s==' '), length(s)+1];
    time = str2int(s(isep(2)+1:isep(3)-1));
    continue;
  end
  warning(sprintf('unrecognized input <%s>', s));
end

if ~isempty(log)
  log.update.counter = log.update.counter(1:updateprop_pos);
  log.update.object = log.update.object(1:updateprop_pos);
  log.update.pos = log.update.pos(1:updateprop_pos);
  log.update.lognumber = log.update.lognumber(1:updateprop_pos);
  log.update.prop = log.update.prop(1:updateprop_pos);
  log.update.prop_value = log.update.prop_value(1:updateprop_pos);
  log.mrk.pos = round(log.mrk.pos*lag);
  log.update.pos = round(log.update.pos*lag);
end

fclose(fid);














function num = str2int(str);
str = double(str)-48;

if any(str<0) | any(str>9)
  str = str(find(str~=-2&str~=-5));
  % kick out '.'
  if any(str==53)
    % e
    ii = find(str==53);
    ex = (10.^(length(str)-ii-1:-1:0))*str(ii+1:end)';
    num = (10.^(ex:-1:ex-ii+2))*str(1:ii-1)';
    return
  else
    error
  end
end

num = (10.^(length(str)-1:-1:0))*str';

