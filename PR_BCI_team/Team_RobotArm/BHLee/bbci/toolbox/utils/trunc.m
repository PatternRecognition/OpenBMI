function vTrunc= trunc(v, digits, policy)
%vTrunc= trunc(v, <digits=4, policy>)
%
% policy is 'floor', 'ceil', or 'round'

if ~exist('digits', 'var'), digits=4; end
if ~exist('policy','var'), policy='round'; end

a= 10^digits;
vTrunc= feval(policy, a*v)/a;
