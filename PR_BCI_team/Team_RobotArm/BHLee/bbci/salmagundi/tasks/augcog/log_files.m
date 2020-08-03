di = '/home/neuro/data/AUGCOG/season_5/';


updrate = 200;
d = dir(di);

ind = [];
for i = 1:length(d)
  if length(d(i).name)>4 & strcmp(d(i).name(end-3:end),'.log')
    ind = [ind,i];
  end
end

d = d(ind);

ty = 'Stimulus';

for fi = 1:length(d);
  fprintf('\r%i/%i   %s ',fi,length(d),d(fi).name);
%for fi = 2;

  mr = [];
  udp = [];
  cl = [];
  fid = fopen([di d(fi).name],'r');
  
  s = '';
  while isempty(strmatch('Start',s)) & ~feof(fid)
    s = fgets(fid);
  end
  if isempty(strmatch('Start',s))
    continue;
  end
  
  s = s(18:end);
  da = datevec(s);

  old = 0;
  while isempty(strmatch('Sampling',s))
    s = fgets(fid);
  end
  s = s(16:end);
  c = strfind(s,' ');
  sr = str2num(s(1:c(1)-1));
    
  time = 0;

  out = struct('fs',sr);
  out.active = [0 1 1]';
  
  while ~feof(fid)
    s = fgets(fid);
    
    if ~isempty(strmatch('Message',s));
      s = s(10:end);
      if ~isempty(strmatch('Got marker',s))
        s = s(13:end);
        c = strfind(s,',');
        typ = s(1:c(1)-1);
        s = s(c(1)+8:end);

        c = strfind(s,',');
        time = str2num(s(1:c(1)-1));
        s = s(c(1)+9:end);
        to = str2num(s);
        
        if strcmp(typ,ty)
          mr = cat(2,mr,[time;to]);
        end
      elseif ~isempty(strmatch('Deactivating',s))
        s = s(25:end);
        s = str2num(s);
        act = out.active(:,end);
        act(1) = time;
        act(s+1) = 0;
        out.active = cat(2,out.active,act);
      elseif ~isempty(strmatch('Activating',s))
        s = s(23:end);
        s = str2num(s);
        act = out.active(:,end);
        act(1) = time;
        act(s+1) = 1;
        out.active = cat(2,out.active,act);
      end
    elseif ~isempty(strmatch('Send_udp',s));
      s = s(16:end);
      c = strfind(s,' ');
      time = str2num(s(1:c(1)-1));
      ud = str2num(s(c(1)+7:end));
      udp = cat(2,udp,[time;ud']);

      
    elseif ~isempty(strmatch('Classifier',s));
      s = s(18:end);
      c = strfind(s,' ');
      time = str2num(s(1:c(1)-1));
      cc = str2num(s(c(1)+7:end));
      cl = cat(2,cl,[time;cc']);
      
    end
  end
  
    
  fclose(fid);

  ncl = size(udp,1)-1;
  if ncl==0 | size(mr,2)==0
    continue;
  end
  
  out.clab = cell(1,2*ncl);
  for i = 1:ncl
    out.clab{i} = sprintf('classifier %i',i);
    out.clab{i+ncl} = sprintf('udp %i',i);
  end
  
  
  
  cl(1,:) = round(cl(1,:)/1000*out.fs);
  udp(1,:) =round(udp(1,:)/1000*out.fs);
  mr(1,:) = round(mr(1,:)/1000*out.fs);
  out.active(1,:) = round(out.active(1,:)/1000*out.fs);
  
  timax = max([out.active(1,end),cl(1,end),udp(1,end),mr(1,end)]);
  
  
  out.x = zeros(2*ncl,timax);

  begin = 1;
  for i = 1:size(cl,2);
    out.x(1:ncl,begin:cl(1,i)) = repmat(cl(2:end,i),[1 cl(1,i)-begin+1]);
    begin = cl(1,i)+1;
  end
  
  begin = 1;
  for i = 1:size(udp,2);
    out.x(ncl+1:2*ncl,begin:udp(1,i)) = repmat(udp(2:end,i),[1 udp(1,i)-begin+1]);
    begin = udp(1,i)+1;
  end
  
  out.pos = mr(1,:);
  out.toe = mr(2,:);
  
  
  
  save([di d(fi).name(1:end-4) '.out'],'out');
  
end

