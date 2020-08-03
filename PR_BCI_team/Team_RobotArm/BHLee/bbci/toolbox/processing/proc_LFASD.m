function [epo,chan] = proc_LFASD(epo,chan,alpha,beta,time,agg)
%PROC_LFASD implements the Mason/Birch LFASD algorithm
%
% usage:
%     [epo,chan] = proc_LFASD(epo,chan,alpha,<beta=0,time=0,agg=0>);
% 
% literature:
%    Mason SG, Birch GE. A brain-controlled switch for asynchronous
%    control applications. IEEE Trans Biomed Eng. 2000
%    Oct;47(10):1297-307.
%
% input:
%    epo:   a usual epo structure
%    chan:  There are several opportunities for given bipolar channels:
%           1. a nx2 array where the differences between the
%              channel (regarding epo.clab) in the first column and
%              the second column is given
%           2. a nx2 cell-array, similar as 1. but the name of the
%              channelstrings are given. It is possible here to use
%              # for 1-10,z or description like 5-6 (from left to right) for
%              more than one channel
%           3. a cell array in the form 'chan-chan' where the
%              strings in chan are given. Also # and 5-6 etc. like
%              in 2 is possible.
%           4. a 2x1 cell array for using different channels for i
%              and j where each cell is of typ 1-3. 
%           5. a nx4 array if you want to use different bipolar channels
%    time:  a vector of time points regarding epo.t the feature
%           vector are chosen to
%    alpha: the number of alpha delays as matrix, vector or number in msec
%    beta:  the number of beta delays as matrix, vector or number (=0) in msec
%    agg:   a number how many msec of the pasts are used for
%           aggregating or an interval in msec how many time around
%           the zero point in epo.t is used.
%
% output:
%    epo    the new epo structure (time(regarding time)xchanxTrials)
%    chan   the input chan is given back in format 1 or 5.
%
% Guido Dornhege, 24/09/2003

if ~exist('agg','var') | isempty(agg)
  agg = 0;
end

if length(agg)==1
  agg = [-agg,0];
end

if ~exist('beta','var') | isempty(beta)
  beta = 0;
end

if size(chan,2)>2
  nx = size(chan,2);
else
  nx = size(chan,1);
end

if size(alpha,2)==1
  alpha = [alpha,alpha];
end

if size(beta,2)==1
  beta = [beta,beta];
end

if iscell(chan)   % format 2 or 3 or 4
  % test on format 4, convert directly to 5
  if size(chan,1)==2 & size(chan,2)==1 & ~ischar(chan{1})
    if ~isnumeric(chan{1})
      chan{1} = convert23to1(epo,chan{1});
    end
    if ~isnumeric(chan{2})
      chan{2} = convert23to1(epo,chan{2});
    end
    
    chan = cat(2,chan{1},chan{2});
  else % convert 2 or 3 to 1
    chan = convert23to1(epo,chan);
  end
end

% now everything is in format 1,5

nChan = size(chan,1);


if size(alpha,1)==1
  alpha = repmat(alpha,[nChan,1]);
end


if size(beta,1)==1
  beta = repmat(beta,[nChan,1]);
end


if ~exist('time','var') | isempty(time)
  time = 0;
end

if size(chan,2)==4
  epo.x = cat(4,epo.x(:,chan(:,1),:)-epo.x(:,chan(:,2),:),epo.x(:, ...
						  chan(:,3),:)-epo.x(:,chan(:,4),:));
  vec = [1,2];
else
 epo.x = epo.x(:,chan(:,1),:)-epo.x(:,chan(:,2),:);
 vec = [1,1];
end

feat = zeros(length(time),nChan,size(epo.x,3));
for i = 1:length(time)
  w = time(i)+agg;
  for j = 1:nChan
    fea = [];
    for k = [1,2]
      wa = w'+alpha(j,k);
      wb = w'+beta(j,k);
      inda = repmat(epo.t,[length(w),1])-repmat(wa,[1, ...
		    length(epo.t)]);
      [dum,inda] = min(abs(inda),[],2);
      indb = repmat(epo.t,[length(w),1])-repmat(wb,[1, ...
		    length(epo.t)]);
      [dum,indb] = min(abs(indb),[],2);
      fea = cat(2,fea,epo.x(inda,j,:,vec(k))-epo.x(indb,j,:,vec(k)));
    end

    fea = (fea(:,1,:)>0&fea(:,2,:)>0).*fea(:,1,:).*fea(:,2,:);
    
    feat(i,j,:) = max(fea,[],1);
  end
end

epo.x = feat;
epo.t = time;
chans = cell(size(chan));
for i = 1:size(chans,1)
  for j = 1:size(chans,2)
    chans{i,j} = epo.clab{chan(i,j)};
  end
end

epo.clab = cell(1,nChan);

for i = 1:nChan
  if size(chan,2)==2
    epo.clab{i} = sprintf('%s-%s',chans{i,1},chans{i,2});
  else
    epo.clab{i} = sprintf('%s-%s*%s-%s',chans{i,1},chans{i,2},chans{i,3},chans{i,4});
  end
end





  %%%%%%%%%%%%%%%%%%%%%%%
  % subfunction
  %%%%%%%%%%%%%%%%%%%%%
  
function chan = convert23to1(epo,chan);
      
    
if size(chan,2)~=2 | (size(chan,1)==1 & ~isempty(strfind(chan{1},'-')))   % format 3, convert to 2
  chans = cell(length(chan),2);
  for i = 1:length(chan)
    c = strfind(chan{i},'-')
    c(i,1) = chan{i}(1:c-1);
    c(i,2) = chan{i}(c+1:end);
  end
  chan = chans;
end

% now we have format 2, convert to 1

chans = [];
ind = [];
for i = 1:size(chan,1)
  in1 = chanind(epo,chan{i,1});
  in2 = chanind(epo,chan{i,2});
  if length(in1)~=length(in2) 
    error('different number of channels are used in differences');
  end
  chans = cat(1,chans,transpose([in1;in2]));
end

chan = chans;