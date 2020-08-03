function [bias,scale,dist] = getClassBiasScale(testTube,outTraces,labels,timevalrest,timevaldescr,timevaldist,dsply);
% GETCLASSBIASSCALE plots a tube regarding dscr and calculates bias and 
% scale to have somewhere no bias, and at some other place equal 
% amplitude for all classes. 
%
% usage:
%    [bias,scale] = getClassBiasScale(testTube,outTraces,labels,timevalrest,timevaldescr,timevaldist);
%
% input:
%    testTube,outTraces,labels: see plot_tubes
%    timevalrest:   a time interval where the tubes for all classes together should be in mean around zero (bias)
%    timevaldescr:  a time interval where the tubes for each class should have the same mean value (scale)
%    
% output:
%    bias:    the optimal bias
%    scale    the optimal scaling for each class
%    dist     the optimal dist (the smallest tubes will be used on timevaldist)
%
% Note the biggest tubes of will be used
%

% Guido Dornhege, 23/03/04

if ~exist('timevalrest','var') | isempty(timevalrest)
  error('You have to specify the 4th argument');
end
if ~exist('timevaldescr','var') | isempty(timevaldescr)
  error('You have to specify the 5th argument');
end
if ~exist('timevaldist','var') | isempty(timevaldist)
  error('You have to specify the 6th argument');
end

x = get(gca,'XLim');
y = get(gca,'YLim');


h = patch([timevalrest,timevalrest([2,1])],[y(1),y(1),0.9*y(1)+0.1*y(2),0.9*y(1)+0.1*y(2)],[0.7,0.7,0.7]);
h = patch([timevaldescr,timevaldescr([2,1])],[y(1),y(1),0.9*y(1)+0.1*y(2),0.9*y(1)+0.1*y(2)],[0.9,0.9,0.9]);
h = patch([timevaldist,timevaldist([2,1])],[y(1),y(1),0.9*y(1)+0.1*y(2),0.9*y(1)+0.1*y(2)],[0.8,0.8,0.8]);


t1 = find(dsply.E>=timevalrest(1) & dsply.E<=timevalrest(2));
t2 = find(dsply.E>=timevaldescr(1) & dsply.E<=timevaldescr(2));
t3 = find(dsply.E>=timevaldist(1) & dsply.E<=timevaldist(2));


outTracrest = outTraces(:,t1,:);
outTracdescr = outTraces(:,t2,:);
outTracdist = outTraces(:,t3,:);
tesTurest = min(testTube(t1,2,:),[],3);
tesTurest = cat(2,tesTurest,max(testTube(t1,end-1,:),[],3));
tesTudist = min(testTube(t3,floor(size(testTube,2)*0.5),:),[],3);
tesTudist = cat(2,tesTudist,max(testTube(t3,ceil(size(testTube,2)*0.5)+1,:),[],3));

tesTudescr = testTube(t2,[2,end-1],:);

clear outTraces testTube val

falses = find(outTracrest<=repmat(tesTurest(:,1)',[1,1,size(outTracrest,3)]) | outTracrest>=repmat(tesTurest(:,2)',[1,1,size(outTracrest,3)]));

N = prod(size(outTracrest))-length(falses);
outTracrest(falses)=0;

bias = -sum(outTracrest(:))/N;




scale = zeros(1,size(tesTudescr,3));

for i = 1:size(tesTudescr,3);
  ind = find(labels(i,:));
  outi = outTracdescr(:,:,ind);
  falses = find(outi<=repmat(tesTudescr(:,1,i)',[1,1,size(outi,3)]) | outi>=repmat(tesTudescr(:,2,i)',[1,1,size(outi,3)]));
  N = prod(size(outi))-length(falses);
  outi(falses)=0;
  scale(i) = sum(outi(:))/N;
end

scale = 1./abs(scale+bias);

falses = find(outTracdist<=repmat(tesTudist(:,1)',[1,1,size(outTracdist,3)]) | outTracdist>=repmat(tesTudist(:,2)',[1,1,size(outTracdist,3)]));

N = prod(size(outTracdist))-length(falses);
outTracdist(falses)=0;


if length(scale)==2
  dist = -sum(outTracdist(find((outTracdist(:)-bias)<0))/scale(1))+sum(outTracdist(find((outTracdist(:)-bias)>0))/scale(2));
else
  dist = (outTracdist-bias).*repmat(scale,[1,size(outTracdist,2),size(outTracdist,3)]);
  dist = sum(dist(:));
end

dist = dist/N;






  
  
  
