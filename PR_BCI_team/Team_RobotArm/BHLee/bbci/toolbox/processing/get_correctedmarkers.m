function mrk = get_correctedmarkers(cnt,mrk,artifacts,criteria);
% GET_CORRECTEDMARKERS gives based on a cnt and mrk structure a
% corrected mrk structure regarding some criteria back
%
% description:
%           mrk = get_correctedmarkers(cnt,mrk,criteria,artifacts);
%
% input: 
%           cnt       usual cnt structure
%           mrk       usual mrk structure
%           criteria  is a struct with 
%                 typ   'gradient' (default) or 'variance', nc-variance'
%                 interval    the interval length in msec where
%                             optimisation takes place (default 50)
%                 range       a interval (regarding onset) where
%                             optimisation is searched, or one value for each
%                             direction from onset (default 500)
%                 channel     the interesting channel
%           criteria can be only typ or nothing.
%           artifacts the name of the artifacts to cutted out
%            
%  
% some default: if artifacts = 'eogh' 'Augen links' and 'Augen
% rechts' are used with criteria.channel = 'eogh' (if this is not
% given), the same for 'eogv' and 'Augen hoch' and 'Augen runter',
% and 'blinks' for 'blinzeln'.
%
% output:
%           mrk       the corrected mrk structure
%
% GUido Dornhege 03.03.02, revised February 2003

if ~exist('artifacts','var') | isempty(artifacts)
  error('not enough input arguments');
end

if ~exist('criteria','var') | isempty(criteria)
  criteria = 'gradient';
end

if ~isstruct(criteria)
  criteria.typ = criteria;
end

if ~iscell(artifacts)
  switch artifacts
   case 'eogv'
    artifacts = {'Augen hoch','Augen runter'};
    if ~isfield(criteria,'channel')
      criteria.channel = 'EOGv';
    end
   case 'eogh'
    artifacts = {'Augen links','Augen rechts'};
    if ~isfield(criteria,'channel')
      criteria.channel = 'EOGh';
    end
   case 'blinks'
    artifacts = {'blinzeln'};
    if ~isfield(criteria,'channel')
      criteria.channel = 'EOGv';
    end
   otherwise
    artifacts = {artifacts};
  end
end

if ~isfield(criteria,'range')
  criteria.range = 500;
end

if length(criteria.range) == 1
  criteria.range = [-criteria.range, criteria.range];
end

if ~isfield(criteria,'typ')
  criteria.typ = 'gradient';
end

if ~isfield(criteria,'interval')
  criteria.interval = 50;
end

ind = chanind(cnt.clab,criteria.channel);

dat = squeeze(cnt.x(:,ind));

relInd = [];

criteria.interval = round(criteria.interval*cnt.fs/1000);
criteria.range = round(criteria.range*cnt.fs/1000);

for i = 1:length(artifacts);
  arti = artifacts{i};
  relInd = [relInd,find(strcmp(mrk.toe,arti))];
end

relInd = sort(relInd);
mrk.pos = mrk.pos(relInd);
mrk.toe = mrk.toe(relInd);


ran = criteria.range(1):criteria.range(2);

for i = 1:length(mrk.pos)
  val = [];
  posi = mrk.pos(i);
  for pos = ran
    data = dat(posi+pos:posi+pos+criteria.interval);
    switch criteria.typ 
     case 'gradient'
      val = [val,max(data)-min(data)];
     case 'ncvariance'
      val = [val, data'*data];
     case 'variance'
      data = data-mean(data);
      val = [val, data'*data];
    end
  end
  [dum,ind] = max(val);
  mrk.pos(i) = mrk.pos(i)+ran(ind);
end

