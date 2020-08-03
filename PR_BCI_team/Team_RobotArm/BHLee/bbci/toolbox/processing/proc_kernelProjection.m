function epo = proc_kernelProjection(epo,w,dat,kernel,varargin);
% CALCULATE w'*kernel(mat,dat,varargin{:}) where mat is matrix
% about nChan*n (permute(epo.x,[2 1 3])(:,:) or cnt.x')
%
% usage: 
%    epo = proc_kernelProjection(epo,w,dat,kernel,params);
%
% input:
%    epo     is epoch or cnt
%    w       projection matrix 
%    dat     reference data
%    kernel  kernel
%    params  additional params
%
% output:
%    epo     transformed epo
% 
% Guido Dornhege, 24/03/03

data = reshape(permute(epo.x,[2 1 3]),[size(epo.x,2),size(epo.x,1)* ...
		    size(epo.x,3)]);

nTrials = size(data,2);   
K = feval(kernel,dat,data,varargin{:});
data = w'*K;

data = reshape(data,[size(w,2),size(epo.x,1),size(epo.x,3)]);
epo.x = permute(data,[2 1 3]);

epo.clab = cell(1,size(w,2));
for i = 1:size(w,2);
  epo.clab{i} = sprintf('Pattern %i',i);
end



% subfunction kernels

function K = poly(dat1,dat2,p,c)
if ~exist('p','var') | isempty(p)
  p = 2;
end

if ~exist('c','var') | isempty(c);
  c = 0;
end

K = (dat1'*dat2+c).^p;



function K = gauss(dat1,dat2,sigma)
if ~exist('sigma','var');
  sigma = 1;
end

n1 = size(dat1,2);
n2 = size(dat2,2);

K = zeros(n1,n2);
if n1>n2
  for i = 1:n2
    da = dat1-repmat(dat2(:,i),[n1,1]);
    da = sum(da.*da,1);
    K(:,i) = exp(-0.5*da'/sigma);
  end
else
  for i = 1:n1
    da = dat2-repmat(dat1(:,i),[n2,1]);
    da = sum(da.*da,1);
    K(i,:) = exp(-0.5*da/sigma);
  end
end

