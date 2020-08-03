function vTrunc= truncsig(v, varargin)
%vTrunc= truncsig(v, <digits=4, policy>)
%
% policy is 'floor', 'ceil', or 'round'

ee= 10.^ceil(log10(abs(v)));
vTrunc= ee*trunc(v/ee, varargin{:});
