function [out] = prep_selectTime(dat, varargin)
% prep_selectTime (Pre-processing procedure):
%
% This function selects the part of a specific time interval
% from continuous or epoched data.
%
% Example:
% out = prep_selectTime(dat, {'Time',[1000 3000]})
%
% Input:
%     dat - Data structure
%     time - Time interval to be selected (ms)
%
% Returns:
%     out - Time selected data structure
%
%
% Seon Min Kim, 04-2016
% seonmin5055@gmail.com


if isempty(varargin)
    warning('Time interval should be specified')
    out = dat;
    return
end
ival = varargin{1}{2};
d = ndims(dat.x);
is = ceil(ival(1)*dat.fs/1000);
ie = floor(ival(2)*dat.fs/1000);

if d == 2
    if ival(1)<0 || ival(2)/1000>size(dat.x,1)/dat.fs
        warning('Selected time interval is out of epoched interval')
        return
    end
    x = dat.x(is:ie,:);
    s = find((dat.t*1000/dat.fs)>=ival(1));
    e = find((dat.t*1000/dat.fs)<=ival(2));
    iv = s(1):e(end);
    t = dat.t(iv);
    y_dec = dat.y_dec(iv);
    y_logic = dat.y_logic(:,iv);
    y_class = dat.y_class(iv);
end
if d == 3
    if ival(1)<dat.ival(1) || ival(2)>dat.ival(end)
        warning('Selected time interval is out of epoched interval')
        return
    end
    iv = [is:ie]-dat.ival(1)*dat.fs/1000+1;
    x = dat.x(iv,:,:);
    time = iv/dat.fs;
end

out = rmfield(dat,'x');
out.x = x;
if isfield(out,'ival')
    out.ival = time;
end
if exist('t','var')
    out.t = t;
    out.y_dec = y_dec;
    out.y_logic = y_logic;
    out.y_class = y_class;
end