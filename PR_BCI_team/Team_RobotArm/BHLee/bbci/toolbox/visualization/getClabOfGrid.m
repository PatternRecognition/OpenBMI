function clab= getClabOfGrid(mnt)
%GETCLABOFGRID - Channel names of channels visible in the grid of a montage
%
%Synopsis:
% clab= getClabOfGrid(mnt)

if isfield(mnt, 'box'),
  idx= find(~isnan(mnt.box(1,:)));
  %% remove index of legend:
  idx(find(idx>length(mnt.clab)))= [];
  clab= mnt.clab(idx);
else
  clab= mnt.clab;
end
