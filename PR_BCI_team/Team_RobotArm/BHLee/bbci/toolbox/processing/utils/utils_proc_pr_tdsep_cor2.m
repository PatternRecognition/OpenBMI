function K=utils_proc_pr_tdsep_cor2(x,tau,mean_flag)

% K=cor2(x,tau,mean_flag)
% computes time delayed correlation matrix using quadratic mean
% x         : data matrix
% tau       : time lag
% mean_flag : 1 == remove the mean ;  0 == do not remove the mean
% 
% version 2.01, 2/14/99 by AZ

%THIS IS UNPUBLISHED PROPRIETARY SOURCE CODE of GMD FIRST Berlin.
%
%  The purpose of this software is the dissemination of
%  scientific work for scientific use. The commercial
%  distribution or use of this source code is prohibited. 
%  (c) 1996-1999 GMD FIRST Berlin, Andreas Ziehe 
%              - All rights reserved -

[m,n] = size(x);

if min(size(x)) == 1
    x = x(:);  
end
 

if nargin < 3,
    mean_flag=1;
end
 
if nargin < 2, 
   tau=0;
end


[m, n] = size(x);

tau=fix(abs(tau));
if tau>m 
   error('Choose tau smaller than Vector size');
end

if mean_flag,
x = x - ones(m,1) * (sum(x)/m); % center the data
end

L=x(1:m-tau,:);   %  time-delayed signal matrix
R=x(1+tau:m,:);

K=L'*R / (m-tau); % compute correlations

K=(K+K')/2;  % symmetrize 

return