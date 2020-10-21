function vTrunc= util_trunc(v, digits, policy)
%vTrunc= util_trunc(v, <digits=4, policy>)
%
% Truncates the number v, digits gives the amount of digits after the
% comma.
%
% policy is 'floor', 'ceil', or 'round'

if ~exist('digits', 'var'), digits=4; end
if ~exist('policy','var'), policy='round'; end

a= 10^digits;
vTrunc= feval(policy, a*v)/a;