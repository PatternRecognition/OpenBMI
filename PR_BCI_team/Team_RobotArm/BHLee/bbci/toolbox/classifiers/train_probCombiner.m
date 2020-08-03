function C = train_probCombiner(dat,labels,dimensions,varargin)
% TRAIN_PROBCOMBINER TRAINS THE COMBINER ALGORITHM PROB
%
% usage: 
%  C = train_probCombiner(dat,labels,dimensions,<lambda1=1,gamma1=0,...);
%
% input:
%  dat     data
%  labels  labels
%  dimensions    cell array of dimensions for each feature
%  lambda1,gamma1 and so on, see train_RDA
%         if only lambda1 and gamma1 given, the values were assumed for all fields.
%
%  output   the classifier

nF = length(dimensions);
if length(varargin)==0
  lambda = ones(1,nF);
  gamma = zeros(1,nF);
else
  lambda = varargin{1}*ones(1,nF);
  if length(varargin)==1
    gamma = zeros(1,nF);
  else
    gamma = varargin{2}*ones(1,nF);
    for i = 2:nF
      if length(varargin)>=2*i-1
        lambda(i) = varargin{i*2-1};
      end
      if length(varargin)>=2*i
        gamma(i) = varargin{i*2};
      end
    end    
  end
end


start = 0;
for i = 1:nF
  x = dat(start+1:start+prod(dimensions{i}),:);
  start = start+prod(dimensions{i});
  if lambda(i) ==1
    CC = train_RLDA(x,labels,gamma(i));
  else
    CC = train_RDA(x,labels,lambda(i),gamma(i));
  end
  if i==1
    C = CC;
  else
    C.w = cat(1,C.w,CC.w);
    C.b = C.b+CC.b;
    if isfield(C,'sq') | isfield(CC,'sq')
      if ~isfield(C,'sq')
        C.sq = zeros(size(C.w,1),size(C.w,1));
      end
      if ~isfield(CC,'sq')
        CC.sq = zeros(size(CC.w,1),size(CC.w,1));
      end
      C.sq = [C.sq,zeros(size(C.sq,1),size(CC.sq,2)); zeros(size(CC.sq,1),size(C.sq,2)),CC.sq];     
    end
  end
end




