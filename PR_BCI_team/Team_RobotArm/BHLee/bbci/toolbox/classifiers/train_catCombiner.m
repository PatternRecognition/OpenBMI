function C = train_catCombiner(dat,labels,dimensions,varargin)
% TRAIN_METACOMBINER TRAINS THE COMBINER ALGORITHM PROB
%
% usage: 
%  C = train_probCombiner(dat,labels,dimensions,<lambda=1,gamma=0>);
%
% input:
%  dat     data
%  labels  labels
%  dimensions    cell array of dimensions for each feature
%  lambda,gamma and so on, see train_RDA
%
%  output   the classifier

if length(varargin)==0
  lambda = 1;
  gamma = 0;
elseif length(varargin)==1
  lambda = varargin{1};
  gamma = 0;
else 
  lambda = varargin{1};
  gamma = varargin{2};
end

if lambda == 1
  C = train_RLDA(dat,labels,gamma);
else
  C = train_RDA(dat,labels,lambda,gamma);
end
