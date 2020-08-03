function epo = getSpecialEvents(file,fs,marker,ival);
%GETSPECIALEVENTS reads out special events of an augcog-file and gives an epo-structure back
%
% usage:
%     epo = getSpecialEvents(file,<fs=100,marker={{'Call','C  1'}, {'Abs','A  1'}, {'Horn','H  1'}},ival=10000>);
%
% input:
%     file      the name of a file
%     fs        the sampling rate
%     marker    a classDef for the special events
%     ival      an interval in msec regarding special events marker, if it is only one number, [0 ival] will be used
%
% output:
%     epo       a usual epo structure
%
% Guido Dornhege, 01/04/2003

if ~exist('ival','var') | isempty(ival)
  ival = 10000;
end

if length(ival)==1
  ival = [0 ival];
end

if ~exist('marker','var') | isempty(marker)
  marker = {{'Call','C  1'}, {'Abs','A  1'}, {'Horn','H  1'}};
end

if ~exist('fs','var') | isempty(fs)
  fs = 100;
end

mrk = readAlternativeMarkers(file,marker);


epo = struct('fs',fs,'x',[],'y',mrk.y);
epo.className = cell(1,length(marker));
for i = 1:length(marker)
  epo.className{i} = marker{i}{1};
end
  
for i = 1:length(mrk.pos);
  cn = readGenericEEG(file, [], fs, mrk.pos(i)+ival);
  epo.x= cat(3,epo.x,cn.x);
end

epo.clab = cn.clab;

  