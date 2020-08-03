function flo = select_feedback(flog,logfi,window,ref);

if nargin<4 | isempty(ref)
  ref = 0;
end
ind = [];
for i = 1:length(flog);
  I = find(~ismember(flog(i).mrk.lognumber,logfi));
  flog(i).mrk.lognumber(I) = [];
  flog(i).mrk.pos(I) = [];
  flog(i).mrk.counter(I) = [];
  flog(i).mrk.toe(I) = [];
  I = find(~ismember(flog(i).update.lognumber,logfi));
  flog(i).update.lognumber(I) = [];
  flog(i).update.counter(I) = [];
  flog(i).update.pos(I) = [];
  flog(i).update.object(I) = [];
  flog(i).update.prop(I) = [];
  flog(i).update.prop_value(I) = [];

  if isempty(flog(i).mrk.lognumber) & isempty(flog(i).update.lognumber)
    ind = [ind,i];
  end
  

end

flog = flog(setdiff(1:length(flog),ind));

flo = struct('start',[]);
flo.fs = flog(1).fs;

flo.init_file = '';
flo.mrk = struct('pos',[],'toe',[],'counter',[],'lognumber',[]);
flo.update = struct('pos',[],'object',[],'counter',[],'lognumber',[],'prop',{{}},'prop_value',{{}});
flo.file = {};
flo.initial = {};

for i = 1:length(flog);
  idx = find(flog(i).update.pos>=window(1) & flog(i).update.pos<=window(2));
  % check if section is longer than five seconds -> skip
  if ~isempty(idx) & flog(i).update.pos(idx(end))-flog(i).update.pos(idx(1))<=5*flo.fs
    continue;
  end
  
  idx = find(flog(i).mrk.pos>=window(1) & flog(i).mrk.pos<=window(2));
  flo.mrk.pos = cat(2,flo.mrk.pos,flog(i).mrk.pos(idx)-ref);
  flo.mrk.counter = cat(2,flo.mrk.counter,flog(i).mrk.counter(idx));
  flo.mrk.toe = cat(2,flo.mrk.toe,flog(i).mrk.toe(idx));
  flo.mrk.lognumber = cat(2,flo.mrk.lognumber,flog(i).mrk.lognumber(idx));
  if length(idx)>0 & isempty(flo.start)
    flo.start = flog(i).start;
  end
  if length(idx)>0 & isempty(flo.init_file) & isfield(flog(i),'init_file')
    flo.init_file = flog(i).init_file;
  end
  if length(idx)>0 
    flo.file = {flo.file{:},flog(i).file};
  end
  idx = find(flog(i).update.pos>=window(1) & flog(i).update.pos<=window(2));
  flo.update.pos = cat(2,flo.update.pos,flog(i).update.pos(idx)-ref);
  flo.update.counter = cat(2,flo.update.counter,flog(i).update.counter(idx));
  flo.update.lognumber = cat(2,flo.update.lognumber,flog(i).update.lognumber(idx));
  flo.update.object = cat(2,flo.update.object,flog(i).update.object(idx));
  flo.update.prop = cat(2,flo.update.prop,flog(i).update.prop(idx));
  flo.update.prop_value = cat(2,flo.update.prop_value,flog(i).update.prop_value(idx));
  
  if length(idx)>0 
    flo.file = {flo.file{:},flog(i).file};
    
    % initial situation
    flo.initial = {flo.initial{:},initial_situation(flog(i).update,idx(1)-1)};
    
    
  end
    

end

flo.file = unique(flo.file);

  
  
return;

function ini = initial_situation(fie,idx);

ini = cell(max(fie.object(1:idx)),2);

for i = 1:idx
  ob = fie.object(i);
  for j = 1:length(fie.prop{i})
    pr = fie.prop{i}{j};
    if isempty(ini{ob,1})
      ini{ob,1} = {pr};
    end
    ind = find(strcmp(ini{ob,1},pr));
    if isempty(ind)
      ini{ob,1} = cat(2,ini{ob,1},{pr});
      ind = length(ini{ob,1});
    end
    ini{ob,2}{ind} = fie.prop_value{i}{j};
  end
end
