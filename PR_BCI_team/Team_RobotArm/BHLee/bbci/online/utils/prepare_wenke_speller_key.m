function [mrk,flogf] = prepare_wenke_speller_key(file,varargin)

if length(varargin)<=0
  fs = 100;
else
  fs = varargin{1};
end

if length(varargin)>1
  flogf = varargin{2};
else
  flogf = load_feedback(file,fs);
end

flogf.mrk.pos = flogf.mrk.counter*40/1000*fs;
flogf.update.pos = flogf.update.counter*40/1000*fs;

if length(flogf)>1
  error('please specify exactly one setup file');
end

[directory,filename] = fileparts(file);

load([directory '/' flogf.file '.mat']);
if isempty(fb_opt.order_sequence)
  warning('something has to be done without ordering');
  return;
end

ind = find(flogf.update.object==16);prop = cell(1,length(ind));
for i = 1:length(ind)
  st = find(strcmp(flogf.update.prop{ind(i)},'String'));
  if ~isempty(st)
    prop{i} = flogf.update.prop_value{ind(i)}{st};
  end
end

for i = length(prop):-1:1
  if isempty(prop{i})
    prop = prop(setdiff(1:length(prop),i));
    ind(i) = [];
  end
end
delay = 0;

mrk = flogf.mrk;
mrk.fs = fs;
classDef = {31,32,60;'left','right','free'};
mrk = makeClassMarkers(mrk,classDef);
pos = 1;
dind = [];
aind = [];
while pos<=length(mrk.toe)
  if mrk.toe(pos)==60 
    if pos == length(mrk.toe) | mrk.toe(pos+1)==60
      dind = [dind,pos];
      pos = pos+1;
    else
      pos = pos+2;
    end
  else
    mrk.pos = [mrk.pos(1:pos-1),NaN,mrk.pos(pos:end)];
    mrk.toe = [mrk.toe(1:pos-1),60,mrk.toe(pos:end)];
    mrk.y = [mrk.y(:,1:pos-1),[0,0,1]',mrk.y(:,pos:end)];
  end
end

mrk.pos(dind) = [];
mrk.toe(dind) = [];
mrk.y(:,dind) = [];
mrkfree = mrk_selectClasses(mrk,'remainclasses','free');
mrk = mrk_selectClasses(mrk,{'left','right'});
mrk.free = mrkfree.pos;


mrk.letter_order = struct('order',{prop},'pos',flogf.update.pos(ind)+delay);

ind = find(flogf.update.object==6);prop = {};col = {};
ind1 = [];
ind2 = [];
for i = 1:length(ind)
  st = find(strcmp(flogf.update.prop{ind(i)},'String'));
  if ~isempty(st)
    prop{end+1} = flogf.update.prop_value{ind(i)}{st};ind1=[ind1,ind(i)];
  end
  st = find(strcmp(flogf.update.prop{ind(i)},'Color'));
  if ~isempty(st)
    col{end+1} = flogf.update.prop_value{ind(i)}{st};ind2=[ind2,ind(i)];
  end
end

ind = ind1;

mrk.buffertext = struct('buffer',{prop},'pos',flogf.update.pos(ind)+delay);
color = struct('color',{col},'pos',flogf.update.pos(ind2)+delay);


ind = 1;
for i = 2:length(mrk.buffertext.buffer)
  if isempty(mrk.buffertext.buffer{i}) 
    if ~isempty(mrk.buffertext.buffer{i-1})
      ind = [ind,i];
    end
  else
    if isempty(mrk.buffertext.buffer{i-1}) | ~strcmp(mrk.buffertext.buffer{i},mrk.buffertext.buffer{i-1})
      ind = [ind,i];
    end
  end
end

mrk.buffertext.buffer = mrk.buffertext.buffer(ind);
mrk.buffertext.pos = mrk.buffertext.pos(ind);


textfield = cell(1,6);
rangind = zeros(1,6);
for ii = 8:13
  ind = find(flogf.update.object==ii);prop = cell(1,length(ind));
  for i = 1:length(ind)
    st = find(strcmp(flogf.update.prop{ind(i)},'String'));
    if ~isempty(st)
      prop{i} = flogf.update.prop_value{ind(i)}{st};
    end
  end
  ind2 = [];
  for i = 1:length(prop)
    if isempty(prop{i})
    elseif strcmp(prop{i},'\_')
      ind2 = [ind2,i];
      prop{i} = 0;
    elseif strcmp(prop{i},'...')
      ind2 = [ind2,i];
      prop{i} = -2;
    elseif strcmp(prop{i},'<')
      ind2 = [ind2,i];
      prop{i} = -1;
    else 
      ind2 = [ind2,i];
      prop{i} = upper(prop{i})-'A'+1;
    end
  end
  
  textfield{ii-7} = struct('text',[prop{ind2}],'pos',flogf.update.pos(ind(ind2))+delay);
  rangind(ii-7) = length(ind2);
end

currind = ones(1,6);

situation = [];


while any(currind<=rangind)
  posis = inf*ones(1,6);
  for ii = 1:6
    if currind(ii)<=rangind(ii)
      posis(ii) = textfield{ii}.pos(currind(ii));
    end
  end
  dum = min(posis);
  ind = find(posis==dum);
  arr = zeros(1,4);
  if ismember(1,ind)
    arr(1) = textfield{1}.text(currind(1));
    arr(2) = textfield{3}.text(currind(3));
  else
    arr(1) = textfield{2}.text(currind(2));
    arr(2) = textfield{2}.text(currind(2));
  end
  if ismember(4,ind)
    arr(3) = textfield{4}.text(currind(4));
    arr(4) = textfield{6}.text(currind(6));
  else
    arr(3) = textfield{5}.text(currind(5));
    arr(4) = textfield{5}.text(currind(5));
  end
  currind(ind) = currind(ind)+1;
  situation = cat(1,situation,[dum,arr]);
end

situation(find(all(diff(situation(:,2:5),1,1)==0,2))+1,:) = [];

ind = [];
for i = 1:length(mrk.pos)
  ind = [ind,max(find(mrk.pos(i)>situation(:,1)))];
end

mrk.situation = situation(ind,:);

forget_text = '';
while mrk.situation(end,1)>mrk.pos(end)
  mrk.situation(end,:) = [];
end

while mrk.situation(1,1)<mrk.letter_order.pos(1)
  mrk.situation(1,:) = [];
end
  
order = zeros(2,size(mrk.situation,1));
ord_old = '';
ord_o = '';

for i = 1:size(mrk.situation,1)
  ind = max(find(mrk.situation(i,1)>=mrk.letter_order.pos));
  ord = mrk.letter_order.order{ind};
  ord(ord=='\') = '';
  if isempty(ord_o)
    ord_o = ord;
  end
  if ~strcmp(ord_o,ord)
    hhh = ord_o;
    ord_o = ord;
    ord = hhh;
  end
    
  ind2 = max(find(mrk.situation(i,1)>=mrk.buffertext.pos));
  if isempty(ind2)
    buf = '';
    forget_text = '';
  else
    buf = mrk.buffertext.buffer{ind2};
  end
    
  buf(buf=='\') = '';
  if fb_opt.begin_text
    fg = forget_text;
    fg(fg=='_')=' ';
    buf(buf=='_')= ' ';
    while ~isempty(fg)
      if strncmp(fg,buf,min(length(fg),length(buf)))
        buf = buf(length(fg)+1:end);
        fg = '';
      else
        fg = fg(2:end);
      end
    end
    if ~isempty(ord_old) & ~strcmp(ord_old,ord)
      if ~isempty(buf),forget_text = [forget_text buf];end
      fg = forget_text;
      fg(fg=='_')=' ';
      buf(buf=='_')= ' ';
      while ~isempty(fg)
        if strncmp(fg,buf,min(length(fg),length(buf)))
          buf = buf(length(fg)+1:end);
          fg = '';
        else
          fg = fg(2:end);
        end
      end
    end
    ord_old = ord;
  end
  
  
  
  
  mist = 0;
  if length(buf)>0,mist = sum(buf~=ord(1:length(buf)));end
  if length(buf)>=length(ord),
    if length(buf)-length(ord)>=fb_opt.tolerance_length | mist<=fb_opt.tolerances_mistakes
      ind = max(find(mrk.situation(i,1)>=mrk.letter_order.pos));
      next_order = mrk.letter_order.order{ind};
      next_order(next_order=='\') = '';
      next_order = next_order(1);
    else
      next_order = '<';
    end
  else
    if mist==0
      next_order = ord(length(buf)+1);
    elseif  mist>fb_opt.tolerance_mistakes
      next_order = '<';
    else
      next_order = ['<',ord(length(buf)+1)];
    end
  end
  goal = map_char(next_order);
  
  for j = 1:length(goal)
    if mrk.situation(i,2)<=goal(j) & mrk.situation(i,3)>=goal(j)
      order(j,i) = 1;
    elseif mrk.situation(i,4)<=goal(j) & mrk.situation(i,5)>=goal(j)
      order(j,i) = 2;
    end
  end
end

if any(order(2,:)),
  mrk.strong_order = cat(1,order(1,:)==1,order(1,:)==2);
  mrk.weak_order = cat(1,order(2,:)==1,order(2,:)==2);
else
  mrk.order = cat(1,order(1,:)==1,order(1,:)==2);
end

mrk.order_pos = mrk.situation(:,1)';

if isfield(mrk,'weak_order')
  mrk.strong_ishit = nan*ones(1,length(mrk.pos));
  ind = find(any(mrk.strong_order,1));
  mrk.strong_ishit(ind) = sum(mrk.y(:,ind).*mrk.strong_order(:,ind),1);

  mrk.weak_ishit = nan*ones(1,length(mrk.pos));
  ind = find(any(mrk.weak_order,1));
  mrk.weak_ishit(ind) = sum(mrk.y(:,ind).*mrk.weak_order(:,ind),1);

  
  mrk.ishit = mrk.strong_ishit;
  ind = find(~isnan(mrk.weak_ishit));
  mrk.ishit(ind) = ((mrk.strong_ishit(ind)+mrk.weak_ishit(ind))>0);
  
else
  mrk.ishit = nan*ones(1,length(mrk.pos));
  ind = find(any(mrk.order,1));
  mrk.ishit(ind) = sum(mrk.y(:,ind).*mrk.order(:,ind),1);
end

if isfield(mrk,'weak_ishit')
  mrk.indexedByEpochs = {'ishit','weak_ishit','situation','strong_ishit','order_pos','weak_order','strong_order','order','free'};
else
  mrk.indexedByEpochs = {'ishit','situation','order_pos','order','free'};
end  

%mrk.ishit = [mrk.ishit, ([1, 1, 0 0]*mrres.y(:,pos2))>0];
return;

function [offset,res] = compare_two_arrays(a,b);

res = inf;
res2 = inf;
for i = 1:floor(a/2)
  fi1 = a(i:min(length(a),length(b)+i-1));
  fi2 = b(1:min(length(b),length(fi1)));
  r = abs(fi1-fi2);
  if sum(r)/length(fi1)<res2
    res = r;
    res2 = sum(res)/length(fi1);
    offset = i-1;
  end
end

c = b;b = a; a = c;
for i = 1:floor(a/2)
  fi1 = a(i:min(length(a),length(b)+i-1));
  fi2 = b(1:min(length(b),length(fi1)));
  r = abs(fi1-fi2);
  if sum(r)/length(fi1)<res2
    res = r;
    res2 = sum(res)/length(fi1);
    offset = 1-i;
  end
end


function a = map_char(b);

a = zeros(1,length(b));

for i = 1:length(b)
  if b(i) == '_'
    a(i) = 0;
  elseif b(i) == '<'
    a(i) = -1;
  else
    a(i) = upper(b(i))-'A'+1;
  end
end
return
