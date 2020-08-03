function blk = blk_selectBlocks(blk,varargin);
%BLK_SELECTBLOCKS will select out of blk specified blocks
%
% usage:
%    blk = blk_selectBlocks(blk,array);
%    blk = blk_selectBlocks(blk,class1,...);
%
% input:
%    blk     - a usual blk structure
%    array   - a vector of ivals regarding blk
%    class1  - a name regarding blk.className, you can use *
%
% output:
%    blk     - a usual blk structure with less ivals regarding this
%              choice and furthermore only with classes remained.
%
% Guido Dornhege, 13/02/04

if ~exist('varargin','var') | isempty(varargin)
  return;
end

if isnumeric(varargin)
  vec = varargin{1};
else
  cl = getClassIndices(blk,varargin{:});
  vec = find(sum(blk.y(cl,:),1));
end

blk.y = blk.y(:,vec);
blk.ival = blk.ival(:,vec);
if isfield(blk, 'pos'),
  blk.pos = blk.pos(:,vec);
end

state= bbci_warning('off', 'mrk');
blk= mrk_removeVoidClasses(blk);
bbci_warning(state);
