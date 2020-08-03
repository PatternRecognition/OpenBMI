function [fv,mrk,why,channel] = proc_setMean(fv,varargin)
% 
% Search for all trials with channels where something stupid
% happened defined by varargin (see find_artifacts). The grand mean
% is set instead at this trials in the specified channel
%
% INPUT: SEE FIND_ARTIFACTS
% OUTPUT: fv as new dat struct and the results given by
% find_artifacts
%
% Guido Dornhege
% 10.07.02


[mrk,why,channel] = find_artifacts(fv,varargin{:});

field = ones(size(fv.x,2),size(fv.x,3));

if isempty(mrk)
  return;
end

for i = 1:length(mrk)
  for j = 1:length(channel{mrk(i)})
    ch = channel{mrk(i)}{j};
    g = find(strcmp(fv.clab,ch));
    field(g,mrk(i)) = 0;
  end
end

dat2 = repmat(reshape(field,[1 size(field)]),[size(fv.x,1),1,1]);
dat = fv.x.*dat2;

if size(fv.y,1) == 1
  fv.y = [fv.y<0; fv.y>0];
end

for i = 1:size(fv.y,1)
  labels = find(fv.y(i,:));
  da = sum(dat(:,:,labels),3);
  gew = sum(field(:,labels),2);
  da = da./repmat(gew',[size(da,1),1]);
  fv.x(:,:,labels) = dat(:,:,labels) + (1-dat2(:,:,labels)).*repmat(da, ...
						  [1 1 length(label)]);
end

labels = find(sum(fv.y,1)==0);
da = sum(dat,3);
gew = sum(field,2);
da = da./repmat(gew',[size(da,1),1]);
fv.x(:,:,labels) = dat(:,:,labels) + ...
    (1-dat2(:,:,labels)).*repmat(da, [1 1 length(labels)]);
