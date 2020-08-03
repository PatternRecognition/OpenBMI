function udp = three2onePostProc(out,varargin);
% THREE2ONEPOSTPROC maps a three dimensional classifier output to one dimension by setting the third class to zero if the algorithm decides to this class. Otherwise the difference between the winner class to the best second class is presented (for class one as negative value, for class two as positive value)
%
% usage:
%    udp = three2onePostProc(out);
%
% input:
%    out    3-dim. column-vector
%
% output:
%    udp    = 0 if out(3)==max(out)
%           = out(2)-max(out([1 3])), if out(2)==max(out)
%           = max(out([2 3]))-out(1), if out(1)==max(out)
%
% Guido Dornhege, 02/02/05
% TODO: extended documentation by Schwaighase
% $Id: three2onePostProc.m,v 1.1 2006/04/27 14:24:59 neuro_cvs Exp $

out = out{1};

if ~all(size(out)==[3 1])
  error('wrong use');
end

[dum,ind] = max(out);

switch ind
 case 1
  udp = max(out([2,3]))-out(1);
 case 2 
  udp = out(2)-max(out([1,3]));
 case 3
  udp = 0;
end


