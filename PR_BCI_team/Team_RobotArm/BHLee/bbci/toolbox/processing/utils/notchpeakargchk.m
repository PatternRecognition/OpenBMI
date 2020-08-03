function [Ab,msg] = notchpeakargchk(Wo,BW,opts)
% NOTCHPEAKARGCHK Validates the inputs for the IIRNOTCH and IIRPEAK
% functions.

%   Author(s): P. Pacheco
%   Copyright 1999-2002 The MathWorks, Inc.
%   $Revision: 1.2 $  $Date: 2001/07/06 17:42:26 $ 

% Define default values.
Ab = abs(10*log10(.5)); % 3-dB width
msg='';

% Check Wo and BW for notch/peak filters.
msg = freq_n_bandwidth(Wo,BW);
if msg, return, end

% Parse and validate optional input args.
[Ab,msg] = parseoptions(Ab,opts);
if msg, return, end

%------------------------------------------------------------------------
function msg = freq_n_bandwidth(Wo,BW)
% Check Wo and BW for notch/peak filters.

msg = '';
% Validate frequency cutoff and bandwidth.
if (Wo<=0) | (Wo >= 1),
    msg = 'The frequency Wo must be within 0 and 1.';
    return;
end

if (BW <= 0) | (BW >= 1),
    msg = 'The bandwidth BW must be within 0 and 1.';
    return;
end

%------------------------------------------------------------------------
function [Ab,msg] = parseoptions(Ab,opts)
% Parse the optional input arguments.

msg='';
if ~isempty(opts),
    [Ab,msg] = checkAtten(opts{1});
end

%------------------------------------------------------------------------
function [Ab,msg] = checkAtten(option)
% Determine if input argument is a scalar numeric value.

% Initialize output args.
Ab = [];
msg = '';
if isnumeric(option) & all(size(option)==1),  % Make sure it's a scalar
	Ab = abs(option);  % Allow - or + values
else
	msg = 'Level of decibels specified by Ab must be a numeric scalar.';
end

% [EOF]
