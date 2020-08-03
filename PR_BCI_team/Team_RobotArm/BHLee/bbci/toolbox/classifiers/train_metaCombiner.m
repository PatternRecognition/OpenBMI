function C = train_metaCombiner(dat,labels,dimensions,varargin)
% TRAIN_METACOMBINER TRAINS THE COMBINER ALGORITHM PROB
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
out = cell(1,nF);
for i = 1:nF
  x = dat(start+1:start+prod(dimensions{i}),:);
  start = start+prod(dimensions{i});
  if lambda(i) ==1
    CC(i) = train_RLDA(x,labels,gamma(i));
  else
    CC(i) = train_RDA(x,labels,lambda(i),gamma(i));
  end
  
  out{i} = apply_separatingHyperplane(CC(i),x);
end

out = cat(1,out{:});

CCC = train_LDA(out,labels);

C = CC(1);
C.w = C.w*CCC.w(1);
C.b = C.b*CCC.w(1);
if isfield(C,'sq')
  C.sq = C.sq*CCC.w(1);
end

for i = 2:nF
  C.w = cat(1,C.w,CC(i).w*CCC.w(i));
  C.b = C.b+CC(i).b*CCC.w(i);
  if isfield(C,'sq') | isfield(CC(i),'sq')
    if ~isfield(C,'sq')
      C.sq = zeros(size(C.w,1),size(C.w,1));
    end
    if ~isfield(CC(i),'sq')
      CC(i).sq = zeros(size(CC(i).w,1),size(CC(i).w,1));
    end
    C.sq = [C.sq,zeros(size(C.sq,1),size(CC(i).sq,2)); zeros(size(CC(i).sq,1),size(C.sq,2)),CC(i).sq*CCC.w(i)];     
  end
end

C.b = C.b+CCC.b;






