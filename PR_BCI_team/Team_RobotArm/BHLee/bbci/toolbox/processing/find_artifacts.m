function [mrk, why, channel] = find_artifacts(dat, channels, criteria, amplitude, maxmin, low)
%[mrk, why] = find_artifacts(dat, channels, criteria)
% or
%[mrk, why] = find_artifacts(dat, channels, gradient, amplitude, maxmin, low)
%
% Find artifacts and gives position back. If why is asked for, the
% reason are given back. If chan is asked for chan are given back
% dat are a struct with .x as datas and .fs as used frequence and
% .clab as known channels.
% channels are the channels where the criteria must be
% right. (cell array) (empty = allChannels)
% criteria can be a struct with fields
%  gradient (value for maximal allowed voltage step)
%  amplitude (struct with field min and max (minimal and maximal
%  allowed amplitude), or a twodimenisional vector)
%  maxmin (value with the maximal allowed absolute difference in a
%  segment)
%  low struct with the two fields activity (the lowest allowed
%  activity) and length (the interval length (not set the whole
%  segment)) or twodimensional vector with this values or one value
%  for the lowest allowed activity
%  In this case amplitude, maxmin, low are only used if this
%  values are not determined in criteria.
% criteria can be a value, then equal to criteria.gradient
%  amplitude is then a struct or a twodimensional vector (equal to
%  criteria.amplitude)
%  maxmin a value (equal to criteria.maxmin)
%  low a struct or a twodimensional vector (equal to criteria.low)
%  or a value


if exist('channels') & ~isempty(channels)
  dat = proc_selectChannels(dat, channels);
end
freq = dat.fs;
chan = dat.clab;
dat = dat.x;

n = nargout;

if ~exist('criteria')
  mrk = [];
  return
end

if ~isstruct(criteria) & ~isempty(criteria)
  criteria.gradient = criteria;
else
  criteria.dummy = [];
end

if ~isfield(criteria,'amplitude') & exist('amplitude') & ~isempty(amplitude)
  criteria.amplitude = amplitude;
end

if ~isfield(criteria,'maxmin') & exist('maxmin') & ~isempty(maxmin)
  criteria.maxmin = maxmin;
end

if ~isfield(criteria,'low') & exist('low') & ~isempty(low)
  criteria.low = low;
end

mrkgr = [];
if isfield(criteria,'gradient')
  gr = dat(2:end,:,:)-dat(1:end-1,:,:);
  gr = abs(gr)>criteria.gradient;
  gro = sum(gr,1);
  gr = sum(gro,2);
  mrkgr = find(gr>0);
  if n > 1
     whygr = cell(size(dat,3),1);
     channelgr = cell(size(dat,3),1);
    
     for i =mrkgr'
       whygr{i,1} = 'High gradient in channels';
       ch = find(gro(1,:,i)>0);
       for j=ch
         whygr{i,1} = [whygr{i,1}, ' ', chan{j}];
       end
       whygr{i,1} = [whygr{i,1}, '\n'];
       channelgr{i} = {chan{ch}};
     end
  end
end

mrkamp = [];
if isfield(criteria,'amplitude')
  if ~isstruct(criteria.amplitude)
     amp = criteria.amplitude;
     criteria.amplitude.min = amp(1);
     criteria.amplitude.max = amp(2);
  end
  mrkmi = [];
  if isfield(criteria.amplitude,'min')
    mi = dat<criteria.amplitude.min;
    mii = sum(mi,1);
    mi = sum(mii,2);
    mrkmi = find(mi>0);
    if n>1
      whymi = cell(size(dat,3),1);
      channelmi = cell(size(dat,3),1);
      for i =mrkmi'
        whymi{i,1} = 'Too minimal amplitude in channels';
        ch = find(mii(1,:,i)>0);
        for j=ch
          whymi{i,1} = [whymi{i,1}, ' ', chan{j}];
        end
        whymi{i,1} = [whymi{i,1}, '\n'];
	channelmi{i} = {chan{ch}};
      end
    end  
 
  end
  mrkma = [];
  if isfield(criteria.amplitude,'max')
    ma = dat>criteria.amplitude.max;
    maa = sum(ma,1);
    ma = sum(maa,2);
    mrkma = find(ma>0);
    if n>1
      whyma = cell(size(dat,3),1);
      channelma = cell(size(dat,3),1);
      for i =mrkma'
        whyma{i,1} = 'Too maximal amplitude in channels';
        ch = find(maa(1,:,i)>0);
        for j=ch
          whyma{i,1} = [whyma{i,1}, ' ', chan{j}];
        end
        whyma{i,1} = [whyma{i,1}, '\n'];
	channelma{i} = {chan{ch}};
      end
    end
   end
  mrkamp = union(mrkma,mrkmi);
end

mrkmaxmin = [];
if isfield(criteria,'maxmin')
  maxmin = max(dat,[],1)-min(dat,[],1);
  maxmini = maxmin>criteria.maxmin;
  maxmin = sum(maxmini,2);
  mrkmaxmin = find(maxmin>0);
  if n > 1
     whymax = cell(size(dat,3),1);
     channelmax = cell(size(dat,3),1);
     for i =mrkmaxmin'
       whymax{i,1} = 'Too maximal difference in channels';
       ch = find(maxmini(1,:,i)>0);
       for j=ch
         whymax{i,1} = [whymax{i,1}, ' ', chan{j}];
       end
       whymax{i,1} = [whymax{i,1}, '\n'];
       channelmax{i} = {chan{ch}};
     end
  end
end

mrklow = [];
if isfield(criteria,'low')
  if ~isstruct(criteria.low)
     lo = criteria.low;
     criteria.low.activity = lo(1);
     if length(lo)>1
     criteria.low.length = lo(2);
     end
  end
  if ~isfield(criteria.low,'length')
      num = size(dat,1);
  else
    num = criteria.low.length *freq/1000;
  end
  if num>=size(dat,1)
    steps = 1;
    num = size(dat,1);
  else
    steps = floor(size(dat,1) - num);
  end
  data= dat;
  for i=1:steps
    lowac = max(data(i:i+num-1,:,:),[],1)-min(data(i:i+num-1,:,:), ...
                                              [],1);
    lowa{i} = lowac < criteria.low.activity;
    lowac = sum(lowa{i});
    mrklow = union(mrklow,find(lowac>0));
  end
  if n>1
    whylow = cell(size(dat,3),1);
    channellow = cell(size(dat,3),1);
    for i =mrklow'
      whylow{i,1} = 'Too low activity in channels';
      ch = [];
      for k =1:steps
        ch = union(ch,find(lowa{k}(1,:,i)>0));
      end
      for j=ch
        whylow{i,1} = [whylow{i,1}, ' ', chan{j}];
      end
      whylow{i,1} = [whylow{i,1}, '\n']; 
      channellow{i} = {chan{ch}};
    end
  end
end
 
 
mrk = union(union(mrkgr,mrklow),union(mrkamp,mrkmaxmin));
if n>1
  why  = cell(size(dat,3),1); 
  for i=1:size(dat,3)
    if exist('whygr')
      why{i,1} = [why{i,1}, whygr{i,1}];
    end
    if exist('whymi')
      why{i,1} = [why{i,1}, whymi{i,1}];
    end
    if exist('whyma')
      why{i,1} = [why{i,1}, whyma{i,1}];
    end
    if exist('whymax')
      why{i,1} = [why{i,1}, whymax{i,1}];
    end
    if exist('whylow')
      why{i,1} = [why{i,1}, whylow{i,1}];
    end
    if ~isempty(why{i,1})
      why{i,1} = sprintf(why{i,1});
    end
  end
end

if n>2
  channel= cell(size(dat,3),1); 
  for i=1:size(dat,3)
    channel{i} = {};
    if exist('channelgr') & ~isempty(channelgr{i})
      channel{i} = {channel{i}{:}, channelgr{i}{:}};
    end
    if exist('channelmi') & ~isempty(channelmi{i})
      channel{i} = {channel{i}{:}, channelmi{i}{:}};
    end
    if exist('channelma') & ~isempty(channelma{i})
      channel{i} = {channel{i}{:}, channelma{i}{:}};
    end
    if exist('channelmax') & ~isempty(channelmax{i})
      channel{i} = {channel{i}{:}, channelmax{i}{:}};
    end
    if exist('channellow') & ~isempty(channellow{i})
      channel{i} = {channel{i}{:}, channellow{i}{:}};
    end
  end
end











