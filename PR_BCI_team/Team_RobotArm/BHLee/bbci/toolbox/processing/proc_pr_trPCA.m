% PROC_PR_TRPCA - task related principal component analysis (trPCA)
%
% Usage:
%   [DAT, W, WINV, C] = proc_pr_trPCA(DAT, NUMCOMP);
%   [DAT, W, WINV, C] = proc_pr_trPCA(DAT, OPTS);
%
% Input:
%   DAT: data structure of epoched data
%   NUMCOMP: number of components per class to be calculated, 
%            (default nChans).
%   OPTS: options structure as output by PROPERTYLIST2STRUCT. Recognized
%         options are:
%         'numComp':  see above.
% Output:
%   DAT: projected data
%   W(c,k,n): transformation matrix (k: components)
%   WINV: pseudoinverse of W
%   C(c1,c2,n): correlation matrix (c1/c2: channels)
%
%
% Description:
% proc_pr_trPCA calculates task related principal component analysis (trPCA)
% for 2 classes.
% please note that you should use the .proc (train/apply) feature in 
% xvalidation. See also proc_csp2 and demos/demo_validation_csp
%
% Example:
%   proc_pr_trPCA(X, 12)
%   proc_pr_trPCA(X, propertylist2struct('numComp',12))
%
function [dat, W, Winv, C] = proc_pr_trPCA(dat,numComp)

  [T, nChans, nEpochs]= size(dat.x);
  nClasses= size(dat.y,1);

  % Standard input argument checking
  error(nargchk(1, 2, nargin));
  
  if nargin<2 | isempty(numComp),
    numComp = nChans;
  end

  % Check whether NUMCOMP has been created by PROPERTYLIST2STRUCT:
  if ispropertystruct(numComp),
    if nargin>2,
      error('With given OPTS, no additional input parameter is allowed');
    end
    % OK, so the second arg was not numComp, but the options structure
    opt = numComp;
    % Set default parameters
    opt = set_defaults(opt, 'numComp', nChans);
    % Extract parameters from options
    numComp = opt.numComp;
  end

  if(numComp>size(dat.x,2))
    error('number of components has to be <= number of channels.');
  end

  if nClasses~=2,
    error('this function works only for 2-class data.');
  end

  for n=1:nClasses
    classInd{n,:} = find(dat.y(n,:));
  end



  % time averaged correlation
  % diagonal terms of epoch averaged matrix
  % exclude comparison of simultaneous epochs from different channels


  C = zeros(nChans,nChans,nClasses);
  for n = 1:nClasses
      si= length(classInd{n});
      for i1 = 1:si-1
         for i2 = i1+1:si
            C(:, :, n) = C(:, :, n) + dat.x(:, :, classInd{n}(i1))'*dat.x(:, :, classInd{n}(i2)) ./ T;
         end
      end
      C(:, :,n) = 2*C(:, :,n) ./ (si.^2 - si);
  end


  % find eigenvectors and compute transformation matrix A

  for n=1:nClasses  

    Csym(:,:,n)= 0.5*C(:,:,n)' + 0.5*C(:,:,n);
    [eigvec,eigval] = eig(Csym(:,:,n));

    diagD= diag(eigval);
  %[dd,di]= sort(min(diagD, 1-diagD),'descend');
    [dd,di]= sort(diagD,'descend');
    fi= di(1:numComp);
    A(:,:,n)=eigvec(:,fi);

  end


  % projection


  Winv= cat(3, pinv(A(:,:,1)), pinv(A(:,:,2)));
  Winv= [Winv(1:numComp,:,1); Winv(1:numComp,:,2)]; 

  W= [A(:,1:numComp,1), A(:,1:numComp,2)];

  dat= proc_linearDerivation(dat, W);




  

