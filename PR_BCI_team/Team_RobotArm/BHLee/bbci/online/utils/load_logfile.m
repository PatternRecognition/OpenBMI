function log = load_logfile(filename,fs);

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

if isdir(filename)
  log = struct([]);
  d = dir([filename '/*.log']);
  for i = 1:length(d);
    fprintf('\rLoading Logfile %s (%d/%d)  ',d(i).name,i,length(d));
    log = cat(1,log,load_logfile([filename '/' d(i).name],fs));
  end
  posi = 1;
  while posi<length(log)
    if isempty(log(posi).segments.time) | isnan(log(posi).segments.time(end,1))
      posi = posi+1;
      continue;
    end
    t2 = datenum(log(posi+1).time)*24*60*60;
    t1 = datenum(log(posi).segments.time(end,:))*24*60*60;
    if t2-t1<2 & strcmp(log(posi).feedback, log(posi+1).feedback)
      aa = combine_logs(log(posi),log(posi+1));
      if ~isempty(aa)
        log(posi) = aa;
        log = cat(1,log(1:posi),log(posi+2:end));
      else
        posi = posi+1;
      end
    else
      posi = posi+1;
    end
  end
  
  return;
end

lengthies = 1000;
log = struct('fs',fs,'filename',filename);
log.time = [];
log.feedback = [];
log.setup_file = [];
log.setup = [];
log.continuous_file_number = [];
log.log_variables = {};
log.mrk = struct('pos',[],'toe',[]);
log.msg = struct('pos',[],'message',{{}});
log.cls = struct('pos',zeros(1,lengthies),'values',NaN*ones(1,lengthies));
log.udp = struct('pos',zeros(1,lengthies),'values',zeros(1,lengthies));
log.changes = struct('pos',[],'time',[],'code',{{}});
log.segments = struct('ival',[],'time',[]);
log.adaptation = struct('pos',[],'time',[],'code',{{}});
log.comment = struct('pos',[],'comment',{{}});

position_cls = 1;
position_udp = 1;

fid = fopen(filename,'r');
ht = [];
while ~feof(fid);
  s = fgets(fid);

  % in some old versions . occurs without meaning in the beginning of some lines
  
  if length(s)>=2 & strcmp(s(1:2),'. ')
    s = s(3:end);
  end
  
  if (strmatch_intern('Classifier out',s))
    s = s(length('Classifier output calculated at timestamp ')+1:end-1);
    ind = find(s==':');
    if position_cls>length(log.cls.pos)
      log.cls.pos = cat(2,log.cls.pos,zeros(1,lengthies));
      log.cls.values = cat(2,log.cls.values,NaN*ones(size(log.cls.values,1),lengthies));
    end
    ht = str2int(s(1:ind(1)-1))*lag;
    if isempty(new_start_time),
      new_start_time = ht;
    end
    log.cls.pos(position_cls) = ht;
    s = s(ind(1)+1:end);
    ss = eval(s); ss = cat(1,ss{:});
    if length(ss)>size(log.cls.values,1)
      log.cls.values = cat(1,log.cls.values,NaN*ones(length(ss)-size(log.cls.values,1),size(log.cls.values,2)));
    end
    log.cls.values(1:length(ss),position_cls) = ss;
    position_cls = position_cls+1;
      
    continue
  end
  if (strmatch_intern('Send to udp',s))
    s = s(length('Send to udp at timestamp ')+1:end-1);
    ind = find(s==':');
    
    if position_udp>length(log.udp.pos)
      log.udp.pos = cat(2,log.udp.pos,zeros(1,lengthies));
      log.udp.values = cat(2,log.udp.values,zeros(size(log.udp.values,1),lengthies));
    end
    ht = str2int(s(1:ind(1)-1))*lag;
    log.udp.pos(position_udp) = ht;
    s = s(ind(1)+2:end);
    % old formats
    if s(1) == '{'
      s = s(2:end);
    end
    ind = strfind(s,',]');
    while ~isempty(ind)
      s = [s(1:ind(1)-1),']',s(ind(1)+2:end)];
      ind = strfind(s,',]');
    end
	s = str2num(s)';
    %%cc: changed for the corrupted logfiles
    if isempty(s)
    	warning('Logging stopped unexpectely: Logfile %s is not complete.',filename);
    	continue	
   	end
   	
    if length(s)>size(log.udp.values,1)
      log.udp.values = cat(1,log.udp.values,zeros(length(s)-size(log.udp.values,1),size(log.udp.values,2)));
    end
    log.udp.values(:,position_udp) = s;

    	
    position_udp = position_udp+1;
    
    continue
  end
  if (strmatch_intern('Writing',s))
    s = s(length('Writing logfile starts at ')+1:end-1);
    log.time = convert_timestring(s);
    new_start_time =[];
    continue
  end
  if (strmatch_intern('Used',s))
    s = s(length('Used feedback: ')+1:end-1);
    log.feedback= s;
    continue
  end
  if (strmatch_intern('Contin',s))
    s = s(length('Continuous file number: ')+1:end-1);
    log.continuous_file_number= str2int(s);
    continue
  end
  if (strmatch_intern('The var',s))
    s = fgets(fid);
    s = s(2:end);
    log.log_variables = str2cell(s);
    continue
  end
  if (strmatch_intern('were s',s))
  	s = [filename(1:end-3) 'mat'];
    log.setup_file= s;
    log.setup = load(log.setup_file);
    continue
  end
  if (strmatch_intern('are s',s))
    s = [filename(1:end-3) 'mat'];
    log.setup_file= s;
    log.setup = load(log.setup_file);
    continue
  end
  if (strmatch_intern('Got a m',s))
    s = s(length('Got a marker at timestamp ')+1:end-1);
    ind = find(s==':');
    ht = str2int(s(1:ind(1)-1))*lag;
    s = s(ind(1)+1:end);
    c = strfind(s,',');
    if isempty(c)
        log.mrk.toe = [log.mrk.toe,str2int(s)];
        log.mrk.pos = [log.mrk.pos,ht];
    else
        s2 = s(c(1)+2:end);
        s = s(1:c(1)-1);
        switch s2
            case 'Response'
                log.mrk.toe = [log.mrk.toe,-str2num(s)];
                log.mrk.pos = [log.mrk.pos,ht];
            case 'Stimulus'
                log.mrk.toe = [log.mrk.toe,str2num(s)];
                log.mrk.pos = [log.mrk.pos,ht];
            case 'Comment'
                log.comment.pos = [log.comment.pos,ht];
                log.comment.comment = {log.comment.comment{:},s};
        end
    end
    continue
  end
  if (strmatch_intern('Parameter',s))
    s = s(length('Parameter changes at timestamp ')+1:end-1);
    ind = find(s=='(');ind = ind(1);
    ht = str2int(s(1:ind-2))*lag;
    log.changes.pos = [log.changes.pos, ht];
    s = s(ind+1:end-1);
    log.changes.time= cat(1,log.changes.time,convert_timestring(s));
    s = fgets(fid);
    log.changes.code = cat(2,log.changes.code,{fgets(fid)});
    continue
  end
  if (strmatch_intern('Classifier adap',s))
    s = s(length('Classifier adapted at timestamp ')+1:end-1);
    ind = find(s=='(');ind = ind(1);
    ht = str2int(s(1:ind-2))*lag;
    log.adaptation.pos = [log.adaptation.pos, ht];
    s = s(ind+1:end-1);
    log.adaptation.time= cat(1,log.adaptation.time,convert_timestring(s));
    log.adaptation.code = {log.adaptation.code{:},fgets(fid)};
    s = fgets(fid);
    continue
  end
  if (strmatch_intern('Message',s))
    s = s(length('Message at timestamp ')+1:end-1);
    ind = find(s==':');
    ht = str2int(s(1:ind(1)-1))*lag;
    log.msg.pos = [log.msg.pos,ht];
    s = s(ind(1)+2:end);
    log.msg.toe = cat(2,log.msg.toe,{s});
    continue
  end
  if (strmatch_intern('The log-file is fi',s))
    s = s(length('The log-file is finished at timestamp ')+1:end-1);
    ind = find(s=='(');ind = ind(1);
    ht = str2int(s(1:ind-2))*lag;
    log.segments.ival = [new_start_time,ht];
    s = s(ind+1:end-1);
    log.segments.time= convert_timestring(s);
    continue
  end
end

if isempty(log.segments.ival) & ~isempty(ht)
  log.segments.ival =  [new_start_time,ht];
  log.segments.time = datevec(datenum(log.time)+ht/log.fs/24/60/60);
end


log.cls.pos = log.cls.pos(1:position_cls-1);
log.cls.values = log.cls.values(:,1:position_cls-1);
log.udp.pos = log.udp.pos(1:position_udp-1);
log.udp.values = log.udp.values(:,1:position_udp-1);












function vec = convert_timestring(str);

vec = zeros(1,6);
ind = find(str=='_');
ind = ind(1);
vec(1) = str2num(str(1:ind-1));
str = str(ind+1:end);
ind = find(str=='_');
ind = ind(1);
vec(2) = str2num(str(1:ind-1));
str = str(ind+1:end);
ind = find(str==',');
ind = ind(1);
vec(3) = str2num(str(1:ind-1));
str = str(ind+2:end);

ind = find(str==':');
ind = ind(1);
vec(4) = str2num(str(1:ind-1));
str = str(ind+1:end);
ind = find(str==':');
ind = ind(1);
vec(5) = str2num(str(1:ind-1));
str = str(ind+1:end);
vec(6) = str2num(str);



function c = str2cell(s);

s = deblank(s);

ind = find(s==' ');
ind = [0,ind,length(s)];

c = cell(1,length(ind)-1);
for i = 1:length(ind)-1
  c{i} = s(ind(i)+1:ind(i+1));
end


function flag = strmatch_intern(str1,str2);

if length(str2)<length(str1)
  flag = false;
  return
end

str1 = double(str1); str2 = double(str2(1:length(str1)));

flag = all(str1==str2);




function num = str2int(str);

str = double(str)-48;

if any(str<0) | any(str>9)
  error
end

num = (10.^(length(str)-1:-1:0))*str';

