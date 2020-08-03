function split_impedances(file,player,goal_file);

cp_hdr = 1;

if nargin==2 | isempty(goal_file)
  goal_file = file;
  cp_hdr = 0 ;
end

global EEG_RAW_DIR

if (isunix & file(1)~='/') | (~isunix & file(2)~=':')
  file = [EEG_RAW_DIR file];
end

file = [file '.vhdr'];

if (isunix & goal_file(1)~='/') | (~isunix & goal_file(2)~=':')
  goal_file = [EEG_RAW_DIR goal_file];
end

goal_file = [goal_file '.vhdr'];

fid = fopen(file,'r');

str = {};

while ~feof(fid);
  str = {str{:},fgets(fid)};
end

fclose(fid);

rind = [];
ind = strmatch('Ch',str);
chan = 0;
for i = 1:length(ind)
  c = strfind(str{ind(i)},'=');
  if ~isempty(c)
    d = strfind(str{ind(i)},',');
    cha = str{ind(i)}(c+1:d(1)-1);
    if (cha(1)=='x' & player==2) 
      str{ind(i)} = sprintf('Ch%d=%s',chan+1,str{ind(i)}(c(1)+2:end));
      chan = chan+1;
    elseif  (cha(1)~='x' & player==1)
      str{ind(i)} = sprintf('Ch%d=%s',chan+1,str{ind(i)}(c(1)+1:end));
      chan = chan+1;
    else
      rind = [rind,ind(i)];
    end
  end
end

ind = strmatch('Channels',str);

stind = ind+3;

ch = 1;
while length(str{stind})>10
  c = strfind(str{stind},' ');
  a = min(find(diff(c)>1));
  c = c(a)+1;
  if str{stind}(c)=='x' & player==2
    str{stind} = sprintf('%d%s%s',ch,repmat(' ',[7-ceil(log10(ch+1)),1]),str{stind}(c+1:end));
    ch = ch+1;
  elseif str{stind}(c)~='x' & player==1
    str{stind} = sprintf('%d%s%s',ch,repmat(' ',[7-ceil(log10(ch+1)),1]),str{stind}(c:end));
    ch = ch+1;
  else
    rind = [rind,stind];
  end
  stind = stind+1;
end

ind = strmatch('Impedance',str);
stind = ind+1;

while stind<=length(str) & length(str{stind})>10
  if str{stind}(1)=='x' & player==2
    str{stind}(1)='';
  elseif str{stind}(1)~='x' & player==1
  else
    rind = [rind,stind];
  end
  stind = stind+1;
end

ind = strmatch('NumberOfChannels',str);
str{ind} = sprintf('NumberOfChannels=%d\n',chan);
ind = strmatch('Number of channels',str);
str{ind} = sprintf('Number of channels: %d\n',chan);

fid = fopen(goal_file,'w');
fprintf(fid,'%s',str{setdiff(1:length(str),rind)});
fclose(fid);

if cp_hdr
  if isunix 
    order = 'cp';
  else
    order = 'copy';
  end
  system([order ' ' file(1:end-5) '.vmrk ' goal_file(1:end-5) '.vmrk']);
  system([order ' ' file(1:end-5) '.eeg ' goal_file(1:end-5) '.eeg']);
end

  
