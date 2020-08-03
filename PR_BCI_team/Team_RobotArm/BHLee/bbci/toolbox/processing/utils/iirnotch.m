function [num,den] = iirnotch(Wo,BW,varargin)
%IIRNOTCH Second-order IIR notch digital filter design.
%   [NUM,DEN] = IIRNOTCH(Wo,BW) designs a second-order notch digital filter
%   with the notch at frequency Wo and a bandwidth of BW at the -3 dB level.
%   Wo must statisfy 0.0 < Wo < 1.0, with 1.0 corresponding to pi 
%   radians/sample.
%
%   The bandwidth BW is related to the Q-factor of a filter by BW = Wo/Q.
%
%   [NUM,DEN] = IIRNOTCH(Wo,BW,Ab) designs a notch filter with a bandwidth
%   of BW at a level -Ab in decibels. If not specified, -Ab defaults to the 
%   -3 dB width (10*log10(1/2)). 
% 
%   EXAMPLE:
%      % Design a filter with a Q-factor of Q=35 to remove a 60 Hz tone from 
%      % system running at 300 Hz.
%      Wo = 60/(300/2);  BW = Wo/35;
%      [b,a] = iirnotch(Wo,BW);  
%      fvtool(b,a);
% 
%   See also IIRPEAK, IIRCOMB, GREMEZ.

%   Author(s): P. Pacheco
%   Copyright 1999-2002 The MathWorks, Inc.
%   $Revision: 1.1 $  $Date: 2001/06/18 19:00:45 $ 

%   References:
%     [1] Sophocles J. Orfanidis, Introduction To Signal Processing
%         Prentice-Hall 1996.

error(nargchk(2,3,nargin));

% Validate input arguments.
[Ab,msg] = notchpeakargchk(Wo,BW,varargin);
error(msg);

% Design a notch filter.
[num,den] = secondorderNotch(Wo,BW,Ab);

%------------------------------------------------------------------------
function [num,den] = secondorderNotch(Wo,BW,Ab)
% Design a 2nd-order notch digital filter.

% Inputs are normalized by pi.
BW = BW*pi;
Wo = Wo*pi;

Gb   = 10^(-Ab/20);
beta = (sqrt(1-Gb.^2)/Gb)*tan(BW/2);
gain = 1/(1+beta);

num  = gain*[1 -2*cos(Wo) 1];
den  = [1 -2*gain*cos(Wo) (2*gain-1)];


% [EOF]