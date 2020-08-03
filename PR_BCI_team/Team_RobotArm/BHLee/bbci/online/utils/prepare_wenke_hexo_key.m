function [mrk,flogf] = prepare_wenke_hexo_key(file,varargin)

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

ind = find(flogf.update.object==64);prop = cell(1,length(ind));
for i = 1:length(ind)
  st = find(strcmp(flogf.update.prop{ind(i)},'String'));
  if ~isempty(st)
    prop{i} = flogf.update.prop_value{ind(i)}{st};
  end
end

delay = 0;

mrk = flogf.mrk;
mrk.fs = fs;
classDef = {11,12,13,14,15,16,21,22,23,24,25,26,31,37;'top step 1','top-right step 1','bottom-right step 1','bottom step 1','bottom-left step 1','bottom-up step 1','top step 2','top-right step 2','bottom-right step 2','bottom step 2','bottom-left step 2','bottom-up step 2','step 1 free','step 2 free'};



mr = makeClassMarkers(mrk,classDef);


mrk = mrk_selectClasses(mr,'remainclasses','*step 1','*step 2');

mrk.letter_order = struct('order',{prop},'pos',flogf.update.pos(ind)+delay);


ind = find(flogf.update.object==63);prop = {};col = {};
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
for i = 1:length(mrk.buffertext.buffer)
  if iscell(mrk.buffertext.buffer{i});
    mrk.buffertext.buffer{i} = mrk.buffertext.buffer{i}{end};
  end
end

ind = find(mrk.buffertext.pos<mrk.pos(1));
mrk.buffertext.buffer(ind) = [];
mrk.buffertext.pos(ind) = [];

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

textfield1 = cell(5,6);
textfield2 = cell(1,6);

for ii = 1:6
  for jj = 1:5
    ind = find(flogf.update.object==ii*7+jj-1);prop = cell(1,length(ind));
    for i = 1:length(ind)
      st = find(strcmp(flogf.update.prop{ind(i)},'String'));
      if ~isempty(st)
        prop{i} = flogf.update.prop_value{ind(i)}{st};
      end
    end
    ind2 = [];
    for i = 1:length(prop)
      if isempty(prop{i})
      elseif strcmp(prop{i},'_')
        ind2 = [ind2,i];
        prop{i} = 0;
      elseif strcmp(prop{i},'?')
        ind2 = [ind2,i];
        prop{i} = 27;
      elseif strcmp(prop{i},'.')
        ind2 = [ind2,i];
        prop{i} = 28;
      elseif strcmp(prop{i},'_')
        ind2 = [ind2,i];
        prop{i} = 29;
      elseif strcmp(prop{i},'<')
        ind2 = [ind2,i];
        prop{i} = -1;
      else 
        ind2 = [ind2,i];
        prop{i} = upper(prop{i})-'A'+1;
      end
    end
    prop = prop(ind2);
    ind = ind(ind2);
    cc = mr.pos(find(mr.y(end-1,:)));
    posi = [];
    for kk = 1:length(ind)
      ind2 = min(find(cc-flogf.update.pos(ind(kk))>0));
      if ~isempty(ind2)
        posi = [posi,cc(ind2)];
      else
        posi = [posi,inf];
      end
    end
    ind  = find(diff(posi)==0);
    posi(ind) = [];
    prop(ind) = [];
    
    ind = find(posi>mrk.pos(end));
    posi(ind) = [];
    prop(ind) = [];
    
    textfield1{jj,ii} = struct('text',[prop{:}],'pos', ...
                                 posi);
  end
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
    elseif strcmp(prop{i},'_')
      ind2 = [ind2,i];
      prop{i} = 0;
    elseif strcmp(prop{i},'?')
      ind2 = [ind2,i];
      prop{i} = 27;
    elseif strcmp(prop{i},'.')
      ind2 = [ind2,i];
      prop{i} = 28;
    elseif strcmp(prop{i},' ')
      ind2 = [ind2,i];
      prop{i} = 29;
    elseif strcmp(prop{i},'<')
      ind2 = [ind2,i];
      prop{i} = -1;
    else 
      ind2 = [ind2,i];
      prop{i} = upper(prop{i})-'A'+1;
    end
  end
  ind = ind(ind2);
  prop = prop(ind2);
  
  cc = mr.pos(find(mr.y(end,:)));
  posi = [];
  for kk = 1:length(ind)
    ind2 = min(find(cc-flogf.update.pos(ind(kk))>0));
    if isempty(ind2)
      posi = [posi,inf];
    else
      posi = [posi,cc(ind2)];
    end
  end
  ind = find(posi>mrk.pos(end));
  posi(ind) = [];
  prop(ind) = [];
  textfield2{1,ii} = struct('text',[prop{:}],'pos', ...
                            posi);
  
end

poi = [1,1];
mrk.situation = struct('pos',[],'hex',{{}});
while poi(1)<=length(textfield1{1,1}.text) | poi(2)<=length(textfield2{1,1}.text)
  if poi(2)>length(textfield2{1,1}.text) |  (poi(1)<=length(textfield1{1,1}.text) & textfield1{1,1}.pos(poi(1))<textfield2{1,1}.pos(poi(2)))
    %step 1
    mrk.situation.pos = [mrk.situation.pos,textfield1{1,1}.pos(poi(1))];
    arr = zeros(5,6);
    for ii = 1:6
      for jj = 1:5
        arr(jj,ii) = textfield1{jj,ii}.text(poi(1));
      end
    end
    mrk.situation.hex = {mrk.situation.hex{:},arr};
    poi(1) = poi(1)+1;
  else
    %step 2
    mrk.situation.pos = [mrk.situation.pos,textfield2{1,1}.pos(poi(2))];
    arr = zeros(1,6);
    for ii = 1:6
      arr(1,ii) = textfield2{1,ii}.text(poi(2));
    end
    mrk.situation.hex = {mrk.situation.hex{:},arr};
    poi(2) = poi(2)+1;
  end
end

ind = [];
dind = [];
for i = 1:length(mrk.pos)
  cc = max(find(mrk.pos(i)>mrk.situation.pos));
  if isempty(cc)
    dind = [dind,i];
  else
      ind = [ind,max(find(mrk.pos(i)>mrk.situation.pos))];
  end
end
mrk.pos(dind) = [];
mrk.toe(dind) = [];
mrk.y(:,dind) = [];
mrk.situation.pos = mrk.situation.pos(ind);
mrk.situation.hex = mrk.situation.hex(ind);
mrk.free = mrk.situation.pos;

  
forget_text = '';
while mrk.situation.pos(end)>mrk.pos(end)
  mrk.situation.pos(end,:) = [];
  mrk.situation.hex(end,:) = [];
end

while mrk.situation.pos(1)<mrk.letter_order.pos(1)
  mrk.situation.pos(1) = [];
  mrk.situation.hex(1) = [];
end
  
order = zeros(2,length(mrk.situation.pos));
ord_old = '';

for i = 1:length(mrk.situation.pos)
  ind = max(find(mrk.situation.pos(i)>=mrk.letter_order.pos));
  ord = mrk.letter_order.order{ind};
  ord(ord=='\') = '';
    
  ind2 = max(find(mrk.situation.pos(i)>=mrk.buffertext.pos));
  if isempty(ind2)
    buf = '';
    forget_text = '';
  else
    buf = mrk.buffertext.buffer{ind2};
  end
  if iscell(buf), buf = buf{1};end
  buf(buf=='\') = '';
  if fb_opt.begin_text
    if ~isempty(ord_old) & ~strcmp(ord_old,ord)
      if ~isempty(buf),forget_text = [buf];end
    end
    forget_text(forget_text=='_')=' ';
    buf(buf=='_')= ' ';
    while length(forget_text)>0 & ~strncmp(forget_text,buf,min(length(forget_text),length(buf)))
      forget_text = forget_text(2:end);
    end
    buf = buf(length(forget_text)+1:end);
    ord_old = ord;
  end
  
  
  
  
  mist = 0;
  if length(buf)>0,
    len = min(length(ord),length(buf));
    mist = sum(buf(1:len)~=ord(1:len));
  end
  if length(buf)>=length(ord),
    if length(buf)-length(ord)>=fb_opt.tolerance_length | mist<=fb_opt.tolerances_mistakes
      ind = max(find(mrk.situation.pos(i)>=mrk.letter_order.pos));
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
    order(j,i) = 0;
    for ll = 1:6
      if ismember(goal,mrk.situation.hex{i}(:,ll))
        order(j,i)=ll;
      end
    end
    if order(j,i)==0 & size(mrk.situation.hex{i},1)==1
      order(j,i)=find(29==mrk.situation.hex{i}(1,:));
    end
  end
  
end

if any(order(2,:)),
  mrk.strong_order = cat(1,order(1,:)==1,order(1,:)==2,order(1,:)==3,order(1,:)==4,order(1,:)==5,order(1,:)==6);
  mrk.weak_order = cat(1,order(2,:)==1,order(2,:)==2,order(2,:)==3,order(2,:)==4,order(2,:)==5,order(2,:)==6);
else
  mrk.order = cat(1,order(1,:)==1,order(1,:)==2,order(1,:)==3,order(1,:)==4,order(1,:)==5,order(1,:)==6);
end

mrk.order_pos = mrk.situation(:,1)';

if isfield(mrk,'weak_order')
  mrk.strong_ishit = (prod((mrk.y(1:6,:)+mrk.y(7:12,:)) == mrk.strong_order,1)==1);
  mrk.weak_ishit = (prod((mrk.y(1:6,:)+mrk.y(7:12,:)) == mrk.weak_order,1)==1);
  
  
  
  mrk.ishit = mrk.strong_ishit;
  ind = find(~isnan(mrk.weak_ishit));
  mrk.ishit(ind) = ((mrk.strong_ishit(ind)+mrk.weak_ishit(ind))>0);
  
else
  mrk.ishit = (prod((mrk.y(1:6,:)+mrk.y(7:12,:)) == mrk.order,1)==1);
end

if isfield(mrk,'weak_ishit')
  mrk.indexedByEpochs = {'ishit','weak_ishit','strong_ishit','order_pos','weak_order','strong_order','order','free'};
else
  mrk.indexedByEpochs = {'ishit','order_pos','order','free'};
end  

ind = find(flogf.update.object==58);dat = zeros(length(ind),7,2);
dind = [];
for i = 1:length(ind)
  st = find(strcmp(flogf.update.prop{ind(i)},'XData'));
  if ~isempty(st)
    dat(i,:,1) = flogf.update.prop_value{ind(i)}{st};
  else
    dind = [dind,i];
  end
  st = find(strcmp(flogf.update.prop{ind(i)},'YData'));
  if ~isempty(st)
    dat(i,:,2) = flogf.update.prop_value{ind(i)}{st};
  end
end

dat(dind,:,:) = [];
ind(dind)=[];

ursprung= [0; -1 + 3*fb_opt.hexradius/2/tan(2*pi/12) + 0.01];

dat(:,:,1) = dat(:,:,1)-ursprung(1);
dat(:,:,2) = dat(:,:,2)-ursprung(2);

dat = squeeze(dat(:,4,:));

mrk.arrow = struct('pos',flogf.update.pos(ind)+delay);
mrk.arrow.length = sqrt(sum(dat.*dat,2))';
     
mrk.arrow.angle = zeros(1,length(mrk.arrow.length));
ind = find(dat(:,1)==0);
nind = find(dat(:,1)~=0);
mrk.arrow.angle(ind) = (dat(ind,2)<0)*pi;
mrk.arrow.angle(nind) = 2*pi*(dat(nind,1)<0)+ sign(dat(nind,1)).*(pi/2-atan(dat(nind,2)./abs(dat(nind,1))));
     
mrk.arrow.info = {'radiant degree in clockwise orientation started at [0;1]'};

mrk.arrow.hex = ceil(mod(mrk.arrow.angle+pi/6,2*pi)*3/pi);

mrk.done = struct('pos',mrk.arrow.pos);
mrk.done.y = zeros(2,length(mrk.arrow.pos));
for i = 2:length(mrk.arrow.pos)
  if mrk.arrow.length(i)-mrk.arrow.length(i-1)>0
    mrk.done.y(2,i) = 1;
  end
  if mrk.arrow.length(i)-mrk.arrow.length(i-1)<0
    mrk.done.y(1,i) = 1;
  end
  if mrk.arrow.angle(i)-mrk.arrow.angle(i-1)>0
    mrk.done.y(1,i) = 1;
  end
  if mrk.arrow.angle(i)-mrk.arrow.angle(i-1)<-pi
    mrk.done.y(1,i) = 1;
  end
end
mrk.done.className = {'turn','grow'};

mrk.desired = struct('pos',mrk.arrow.pos);
mrk.desired.className = {'turn','grow'};
mrk.desired.y = zeros(2,length(mrk.arrow.pos));

for i = 1:length(mrk.order_pos.pos)
  if i<length(mrk.order_pos.pos)
    ind = find(mrk.desired.pos>=mrk.order_pos.pos(i) & mrk.desired.pos<mrk.order_pos.pos(i+1));
  else
    ind = find(mrk.desired.pos>=mrk.order_pos.pos(i));
  end    
  if isfield(mrk,'order')
    g = find(mrk.order(:,i));
    if isempty(g)
      g = nan;
    end
  
    mrk.desired.y(1,ind) = (g~=mrk.arrow.hex(ind));
    mrk.desired.y(2,ind) = (g==mrk.arrow.hex(ind));
  else
    mrk.weak_desired = mrk.desired;
    g = find(mrk.strong_order(:,i));
    if isempty(g)
      g = nan;
    end
  
    mrk.desired.y(1,ind) = (g~=mrk.arrow.hex(ind));
    mrk.desired.y(2,ind) = (g==mrk.arrow.hex(ind));
    g = find(mrk.weak_order(:,i));
    if isempty(g)
      g = nan;
    end
  
    mrk.weak_desired.y(1,ind) = (g~=mrk.arrow.hex(ind));
    mrk.weak_desired.y(2,ind) = (g==mrk.arrow.hex(ind));
  end    
end

if ~isfield(mrk,'order')
  mrk.strong_desired = mrk.desired;
  mrk = rmfield(mrk,'desired');
end
  
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
  elseif b(i) == '?'
    a(i) = 27;
  elseif b(i)=='.'
    a(i) = 28;
  elseif b(i)==' '
    a(i) = 29;
  else
    a(i) = upper(b(i))-'A'+1;
  end
end
return



