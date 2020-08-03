function dat = setTrainset(fv, idxTr)
%fv= setTrainset(fv, idx) 
%
% Unformally speaking this function returns fv(idx).
% More specifically the subindexing is performed on the last dimension
% of the fields .x and .y and (if existing) on the fields .bidx, .jit.
% If fv is a cell array of fv structs setTrainset operates on all
% cell elements. This is for needed for feature combination.
%
% DUDU
% 01.07.02

if iscell(fv),
  for j = 1:length(fv)
    dat{j} = setTrainset(fv{j},idxTr);
  end
  return;
end

if ~isstruct(fv),
  sz = size(fv);
  dat = getfield(reshape(fv, [prod(sz(1:end-1)),sz(end)]), ...
                 {1:prod(sz(1:end-1)), idxTr});
  dat = reshape(dat, [sz(1:end-1), length(idxTr)]);
  return;
end

dat = copy_struct(fv, 'not', 'x');
dat.x = setTrainset(fv.x,idxTr);
if isfield(fv,'nx')
   dat.nx = setTrainset(fv.nx,idxTr);
end
if isfield(fv,'y')
  dat.y = fv.y(:,idxTr);
end
if isfield(fv,'bidx')
  dat.bidx = fv.bidx(idxTr);
end
if isfield(fv,'jit')
  dat.jit = fv.jit(idxTr);
end
