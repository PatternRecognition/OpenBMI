function [fv, opt, true_model]= generate_gaussian_toydata(nTrials, varargin)
%GENERATE_GAUSSIAN_TOYDATA - Generates toydata (two Gaussian distributions)
%
%Synopsis:
% [FV, OPT, TRUE_MODEL]= generate_gaussian_toydata(NTRIALS, <OPT, ...>)
%
%Input:
% NTRIALS: a vector of length 2, specifying the number of trials for the
%  two distributions. When NTRIALS is a scalar, each class will have that
%  number of trials.
%
% OPT: struct or property/value list of optional properties:
%  .nInfo: number of informative dimensions. Default Value 100.
%  .stdInfo: upper bound for the standard deviation of the informative
%     dimensions. When .stdInfo is a vector of length 2, the upper bound for
%     std is linear increasing from the first to the second value.
%     Default is [0.5 50].
%  .equilizeCov: scalar in [0-1] used to morph between individual covariance
%     matrices (0) and a common covariance matrix (1). Default Value 0.5.
%  .meanshiftPolicy: determines the offset between the two distributions. Can be
%     one of 'fisher', 'r-value', 'r-square', 't-scale'. In each
%     of the original informative dimensions the meanshift is selected such
%     that the value of the chosen measure ('fisher', ...) matches 
%     OPT.meanshiftParam. Default is 'r-value'.
%  .meanshiftParam parameter for OPT.meanshiftPolicy. Default value is 0.25.
%  .nNoise: number of noise dimensions. Default value 150.
%  .stdNoise: like .stdInfo, but for the noise dimensions. Default is [10 100].
%  .nProj: number of dimensions into which informative plus noise dimensions
%     are rotated.
%  .nNuisance: number of noise dimensions which are added after rotation.
%  .stdNuisance: like .stdInfo, but for the nuisance dimensions.
%     Default value is 50.
%
%Output:
% FV: structure of feature vectors in the toolbox format
%  (as required by, e.g, xvalidation)
% OPT: structure of properties which specified the generation of the
%  distributions (and also the state of the random generators)
% TRUE_MODEL: true mean and covariance matrices that have been used
%  to generate the data
%
%See also: xvalidation.

% Author(s): Benjamin Blankertz, May 2005

if length(nTrials)==1,
  nTrials= [nTrials nTrials];
else
  if length(nTrials)>2,
    error(sprint('%s if so far implemented for two classes only', mfilename));
  end
end

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, ...
                  'nInfo', 100, ...
                  'nNoise', 100, ...
                  'nProj', [], ...
                  'nGivens', '(opt.nInfo+opt.nNoise)^1.75/4', ...
                  'nNuisance', 0, ...
                  'equilizeCov', 0.2, ...
                  'stdInfo', [0.5 50], ...
                  'stdNoise', [10 100], ...
                  'stdNuisance', 50, ...
                  'meanshiftPolicy', 'r-value', ...
                  'meanshiftParam', 0.25);

if isempty(opt.nProj),
  opt.nProj= opt.nInfo + opt.nNoise;
end
if length(opt.stdInfo)==2,
  opt.stdInfo= linspace(opt.stdInfo(1), opt.stdInfo(2), opt.nInfo)';
end
if length(opt.stdNoise)==2,
  opt.stdNoise= linspace(opt.stdNoise(1), opt.stdNoise(2), opt.nNoise)';
end
if length(opt.stdNuisance)==2,
  opt.stdNuisance= linspace(opt.stdNuisance(1), opt.stdNuisance(2), ...
                            opt.nNuisance)';
end

opt.rand_state= rand('state');
opt.randn_state= randn('state');

fv= struct('y', [[1;0]*ones(1,nTrials(1)), [0;1]*ones(1,nTrials(2))]);
fv.className= {'gaussian1', 'gaussian2'};

if sum(nTrials)==0,
  fv.x= zeros(opt.nProj+opt.nNuisance, 0);
  return;
end


%% generate informative dimensions
x1= randn(opt.nInfo, nTrials(1));
x2= randn(opt.nInfo, nTrials(2));
if isfield(opt, 'nonlinFactor'),
  %% this part is included for historic reasons
  stdInfo2= opt.stdInfo + opt.nonlinFactor*opt.stdInfo.*randn([opt.nInfo 1]);
  stdInfo2= abs(stdInfo2);
  dInfo1= opt.stdInfo .* rand([opt.nInfo 1]);
  dInfo2= dInfo1 + stdInfo2.*rand([opt.nInfo 1]);
else
  cInfo1= opt.stdInfo .* rand([opt.nInfo 1]);
  cInfo2= opt.stdInfo .* rand([opt.nInfo 1]);
  perm= randperm(opt.nInfo);
  cInfo1= cInfo1(perm);
  perm= randperm(opt.nInfo);
  cInfo2= cInfo2(perm);
  cInfoPooled= mean([cInfo1 cInfo2], 2);
  dInfo1= (1-opt.equilizeCov)*cInfo1 + opt.equilizeCov*cInfoPooled;
  dInfo2= (1-opt.equilizeCov)*cInfo2 + opt.equilizeCov*cInfoPooled;
end
x1= diag(sqrt(dInfo1))*x1;
x2= diag(sqrt(dInfo2))*x2;

%% determine how to shift the second distribution
switch(lower(opt.meanshiftPolicy)),
 case 'r-value',
  den= std([x1, x2], [], 2);
  fac= sqrt(prod(nTrials))/sum(nTrials);
  meanShift= opt.meanshiftParam * den / fac;
 case 'r-square',
  den= std([x1, x2], [], 2);
  fac= sqrt(prod(nTrials))/sum(nTrials);
  meanShift= sqrt(opt.meanshiftParam) * den / fac;
 case 'fisher',
  den= var(x1')' + var(x2')';
  meanShift= sqrt( opt.meanshiftParam * den );
 case 't-scale',
  den= sqrt( ( (nTrials(1)-1)*var(x1')' + (nTrials(2)-1)*var(x2')' ) ...
       * (1/nTrials(1)+1/nTrials(2)) / (sum(nTrials)-2) );
  meanShift= opt.meanshiftParam * den;
 case 'minstd',
  meanShift= opt.meanshiftParam * min(std(x1'), std(x2'))';
 otherwise,
  error('unknown meanshiftPolicy');
end
x2= x2 + meanShift*ones([1 nTrials(2)]);

X= [x1, x2];
true_model.classMean= [zeros(opt.nInfo, 1), meanShift];
true_model.classCov= cat(3, diag(dInfo1), diag(dInfo2));

%if size(X,1)==2,
%  plot(X(1,1:nTrials(1)), X(2,1:nTrials(1)),'.'); hold on;
%  plot(X(1,nTrials(1)+[0:nTrials(2)-1]), X(2,nTrials(1)+[0:nTrials(2)-1]), ...
%       'r.'); hold off;
%  axis square equal
%end


%% generate noise dimensions and append them to the informative ones
dNoise= opt.stdNoise .* rand([opt.nNoise 1]);
cNoise= diag(dNoise);
X= [X; diag(sqrt(dNoise)) * randn(opt.nNoise, sum(nTrials))];
true_model.classMean= cat(1, true_model.classMean, zeros(opt.nNoise, 2));
cov_class1= [true_model.classCov(:,:,1), zeros(opt.nInfo, opt.nNoise); 
             zeros(opt.nNoise, opt.nInfo), cNoise];
cov_class2= [true_model.classCov(:,:,2), zeros(opt.nInfo, opt.nNoise); 
             zeros(opt.nNoise, opt.nInfo), cNoise];
true_model.classCov= cat(3, cov_class1, cov_class2);


%%%%% ------ begin: Old stupid code
%% rotate informative and noise dimensions into (a possibly higher
%% dimensional) space
%% With the following loop I try to exclude bad projection matrices.
%% Probably this is not necessary, since an orthonromal basis is choose
%% afterwards.
ok= 0;
while ~ok,
  P= randn([opt.nProj, opt.nInfo+opt.nNoise]);
  s= svd(P);
  ok= max(s)/min(s) < 2*(opt.nInfo+opt.nNoise);
end
oP= orth(P);
%%%%% ------ end: Old stupid code

%% --- begin: New code - in construction
% $$$ nGivens= ceil(eval(opt.nGivens));
% $$$ %keyboard
% $$$ nDimOut= opt.nProj;
% $$$ nDimIn= opt.nInfo+opt.nNoise;
% $$$ oP= eye(nDimOut, nDimIn);
% $$$ [so,si]= sort(diag(mean(true_model.classCov,3)));
% $$$ i1_list= si(1:predef);
% $$$ i2_list= si(end-predef+1:end);
% $$$ for gg= 1:nGivens,
% $$$   th= rand*2*pi;
% $$$   c= cos(th);
% $$$   s= sin(th);
% $$$   if gg<predef,
% $$$     i1= i1_list(gg);
% $$$     i2= i2_list(gg);
% $$$   else
% $$$     i1= ceil(nDimOut*rand);
% $$$     i2= ceil(nDimOut*rand);
% $$$   end
% $$$   R= eye(nDimOut);
% $$$   R([i1 i2],[i1 i2])= [c s; -s c];
% $$$   oP= R*oP;
% $$$ end
%% --- end: New code - in construction

X= oP*X;
true_model.classMean= oP*true_model.classMean;
cov_class1= oP*true_model.classCov(:,:,1)*oP';
cov_class2= oP*true_model.classCov(:,:,2)*oP';
true_model.classCov= cat(3, cov_class1, cov_class2);


%% generate noise dimensions and append them to the data
if opt.nNuisance>0,
  dNuisance= opt.stdNuisance .* rand([opt.nNuisance 1]);
  cNuisance= diag(dNuisance);
  X= [X; diag(sqrt(dNuisance)) * randn(opt.nNuisance, sum(nTrials))];
  true_model.classMean= [true_model.classMean; zeros(opt.nNuisance, 2)];
  cov_class1= [true_model.classCov(:,:,1), zeros(opt.nProj, opt.nNuisance); 
               zeros(opt.nNuisance, opt.nProj), cNuisance];
  cov_class2= [true_model.classCov(:,:,2), zeros(opt.nProj, opt.nNuisance); 
             zeros(opt.nNuisance, opt.nProj), cNuisance];
  true_model.classCov= cat(3, cov_class1, cov_class2);
end

fv.x= X;
