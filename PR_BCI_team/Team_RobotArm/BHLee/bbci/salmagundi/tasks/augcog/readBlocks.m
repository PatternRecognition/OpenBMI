function [cnt,mrk,mrk2] = readBlocks(file,blk,classDef,cont);
%READBLOCKS reads out blk to give cnt and mrk mrk2 structure
%
% usage:
%   [cnt,mrk,mrk2,mnt] = readBlocks(file,blk,classDef,cont);
%   [cnt,mrk,mrk2,mnt] = readBlocks(blk,classDef,cont); % if the file information is
%                                             %provided in blk
%
% input:
%   file    a filename
%   blk     a usual blk structure
%   classDef  a classDef for getting mrk2
%   cont    a flag (default false) if true the cnt is continuously provided 
%
% output:
%   cnt     a usual cnt structure
%   mrk     a usual mrk structure (with start markers)
%   mrk2    a usual mrk structure (regarding classDef)
%
% Guido Dornhege, 13/02/2004

if ~exist('classDef','var') | isempty(classDef)
  classDef = [];
end

if ~exist('cont','var') | isempty(cont)
  cont = false;
end

if isstruct(file)
  if ~exist('blk','var') | isempty(blk)
    blk = [];
  end
  cont = classDef; % this line added by mikio
  classDef = blk;
  blk = file;
  file = blk.name;
end

if isempty(classDef)
  classDef = {'D*','S*';'D','S'};
end


ivi = blk.ival;
if cont
  ivi = [ivi;[ivi(1,2:end),ivi(2,end)]];
end

%iv = [blk.ival(1,1);blk.ival(2,end)];
iv = round(ivi*1000/blk.fs);

state= bbci_warning('off', 'mrk');
for i = 1:size(iv,2);
  cn= readGenericEEG(blk.name, [], blk.fs, iv(1,i), iv(end,i)-iv(1,i)+1000/blk.fs);
  cl = blk.className{find(blk.y(:,i))};
  mr = struct('pos',[1,ivi(2,i)-ivi(1,i)+1],'fs',blk.fs,'y',1,'className',{{cl}});
  mr2 = read_marker_classes(blk.name, blk.fs, classDef, iv(1:2,i)');
  for j = 1:length(mr2.className)
    mr2.className{j} = [mr2.className{j} ' ' cl];
  end
  if i==1,
    cnt = cn;
    mrk = mr;
    mrk2 = mr2;
  else
    mr2.pos = mr2.pos + size(cnt.x,1);
    [cnt,mrk] = proc_appendCnt(cnt,cn,mrk,mr);
    mrk2 = mrk_mergeMarkers(mrk2,mr2);
  end
end
bbci_warning(state);
 
if size(iv,2)==0
  mrk2 = struct([]);
  mrk = struct([]);
  cnt = struct([]);
else
  [mrk2.pos,ind] = sort(round(mrk2.pos));
  mrk2.y = mrk2.y(:,ind);
  mrk.end = size(cnt.x,1);
end

endpos = mrk.pos(2:2:end);
mrk.pos = mrk.pos(1:2:end);
mrk.ival= [mrk.pos; endpos];
[dum,mrk.toe] = max(mrk.y,[],1);
mrk.toe = cellstr(num2str(mrk.toe'))';

mrk.indexedByEpochs = {'ival'};