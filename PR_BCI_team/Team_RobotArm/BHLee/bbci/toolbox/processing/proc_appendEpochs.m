function epo= proc_appendEpochs(epo, epo_append, mrk, mrk_append)
%epo= proc_appendEpochs(epo, epo_append, <mrk, mrk_append>)
%epo= proc_appendEpochs(epo_cell, <mrk_cell>)
%
% appends the epochs of 'epo_append' to 'epo'
% if 'epo' is empty 'epo_append' is returned
%
% if mrk structures are given, epochs are sorted chronologically
% does NOT work for jittered epochs!!!
%
% SEE  makeEpochs

if iscell(epo),
  Cepo= epo;
  epo= Cepo{1};
  if nargin>1,
    Cmrk= mrk;
    mrk= Cmrk{1};
    for ii= 2:length(Cepo),
      [epo, mrk]= proc_appendEpochs(epo, Cepo{ii}, mrk, Cmrk{ii});
    end
  else
    for ii= 2:length(Cepo),
      epo= proc_appendEpochs(epo, Cepo{ii});
    end
  end
  return
end
  
    
if isempty(epo),
  epo= epo_append;
  return;
end

% begin sthf
if isfield(epo, 'ndims')
  ndims = max(3, epo.ndims);
else
  ndims = max(3, length(size(epo.x)));
end

if size(epo.x,ndims-2)~=size(epo_append.x,ndims-2),
  error('interval length mismatch');
end
if size(epo.x,ndims-1)~=size(epo_append.x,ndims-1),
  error('number of channels mismatch');
end
if ndims == 4
  if size(epo.x,ndims-3)~=size(epo_append.x,ndims-3),
    error('number of frequencies mismatch');
  end
end

epo.x= cat(ndims, epo.x, epo_append.x);
if isfield(epo, 'p') && isfield(epo_append, 'p') % if epo is r-value
  epo.p= cat(ndims, epo.p, epo_append.p);
end
if isfield(epo, 'V') && isfield(epo_append, 'V') % if epo is r-value
  epo.V= cat(ndims, epo.V, epo_append.V);
end
% end sthf

epo.y= cat(2, epo.y, zeros(size(epo.y,1),size(epo_append.y,2)));
fie = {};

if isfield(epo, 'indexedByEpochs') & isfield(epo_append, 'indexedByEpochs'),
  idxFields= intersect(epo.indexedByEpochs, epo_append.indexedByEpochs);
  for Fld= idxFields,
    fld= Fld{1};
    tmp= getfield(epo, fld);
    sz= size(tmp);
    fie = {fie{:},fld};
    eval(sprintf('epo.%s= cat(length(sz), tmp, epo_append.%s);', ...
                 fld, fld));
  end
end

if sum(strcmp(fie,'jit'))==0 & (isfield(epo, 'jit') | isfield(epo_append, 'jit')),
  if ~isfield(epo, 'jit'), 
    epo.jit= zeros(1, size(epo.y,2));
  end
  if ~isfield(epo_append, 'jit'), 
    epo_append.jit= zeros(1, size(epo_append.y,2));
  end
  epo.jit= cat(2, epo.jit, epo_append.jit);
end

if sum(strcmp(fie,'bidx'))==0 & (isfield(epo, 'bidx') | isfield(epo_append, 'bidx')),
  if ~isfield(epo, 'bidx'), 
    epo.bidx= 1:size(epo.y,2); 
  end
  if ~isfield(epo_append, 'bidx'), 
    epo_append.bidx= size(epo.y,2)+1:size(epo.y,2)+size(epo_append.y,2); 
  end
  epo.bidx= cat(2, epo.bidx, epo_append.bidx);
end
  
if isfield(epo, 'className') & isfield(epo_append, 'className'),
  for i = 1:length(epo_append.className)
    c = find(strcmp(epo.className,epo_append.className{i}));
    if isempty(c)  
      epo.y= cat(1, epo.y, zeros(1,size(epo.y,2)));
      epo.className=  cat(2, epo.className, {epo_append.className{i}});
      c= size(epo.y,1);
    elseif length(c)>1,
      error('multiple classes have the same name');
    end
    epo.y(c,end-size(epo_append.y,2)+1:end) = epo_append.y(i,:);
  end
end

if exist('mrk_append', 'var'),
  [si,si]= sort([mrk.pos+epo.t(end) mrk_append.pos+epo_append.t(end)]);
  if isfield(epo, 'bidx'),
    error('not implemented');
  end
% begin sthf
  subind ='';
  for isi = 1:ndims-1
    subind = [subind ':,'];
  end
  epo.x= eval(['epo.x(' subind 'si);']);
% end sthf  
  epo.y= epo.y(:,si);
end
