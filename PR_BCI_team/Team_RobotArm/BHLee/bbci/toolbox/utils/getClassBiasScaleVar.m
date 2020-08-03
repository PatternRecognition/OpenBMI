function [bias,scale,dist] = getClassBiasScaleVar(testTube,outTraces,labels,timeval,equalvars,dsply);
% GETCLASSBIASSCALEVAR plots a tube regarding dscr and calculates bias and 
% scale to have a good bias and scaling
%
% usage:
%    [bias,scale] = getClassBiasScale(testTube,outTraces,labels,timeval,equalvars);
%
% input:
%    testTube,outTraces,labels: see plot_tubes
%    timeval:   a time interval where you look on
%    equalvars: a values between 0 and 1 (to get equal vars or not)
%    
% output:
%    bias:    the optimal bias
%    scale    the optimal scaling for each class
%    dist     the optimal dist (the smallest tubes will be used on timevaldist)
%
% Note the biggest tubes of will be used
%

% Guido Dornhege, 23/03/04

if ~exist('timeval','var') | isempty(timeval)
  error('You have to specify the 4th argument');
end

if ~exist('equalvars','var') | isempty(equalvars);
  equalvars = 0;
end


if size(labels,1)~=2
  error('only works for 2 classes');
end

x = get(gca,'XLim');
y = get(gca,'YLim');


h = patch([timeval,timeval([2,1])],[y(1),y(1),0.9*y(1)+0.1*y(2),0.9*y(1)+0.1*y(2)],[0.7,0.7,0.7]);


t1 = find(dsply.E>=timeval(1) & dsply.E<=timeval(2));


outTraces = outTraces(:,t1,:);

testTube = testTube(:,[2,floor(size(testTube,2)*0.5),1+ceil(size(testTube,2)*0.5),end-1],:);

testTube = testTube(t1,:,:);

a(1) = squeeze(mean(testTube(:,3,1),1));
a(2) = squeeze(mean(testTube(:,2,2),1));

s = squeeze(mean(testTube(:,3,:)-testTube(:,2,:),1));


w1 = (1-equalvars)*s(2)+(1+equalvars)*s(1);
w1 = 2*(s(1)+s(2))/(a(2)-a(1))/w1;

w2 = a(2)-a(1)-1/w1;
w2 = 1/w2;
b = 1/w2-a(2);

% $$$ w1 = (1+equalvars)*(a(2)*s(1)-s(2)*a(1))+(1-equalvars)*(s(2)*a(2)-s(1)*a(1));
% $$$ w1 = (2*(1-equalvars)*s(1)+2*(1+equalvars)*s(2))/w1;
% $$$ 
% $$$ w2 = (2+w1*a(1))/a(2);
% $$$ 
% $$$ b = 1-w2*a(2);

bias = b;
scale = [w1,w2];

d= zeros(2,1);
d(1) = squeeze(mean(testTube(:,4,1),1));
d(2) = squeeze(mean(testTube(:,1,2),1));

dist = mean(abs(scale*(d+b)));





  
  
  
