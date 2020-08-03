function delete_hnd(h)

if iscell(h),
  for ii= 1:length(h),
    delete_hnd(h{ii});
  end
elseif isstruct(h),
  flds= fieldnames(h);
  for ii= 1:length(flds),
    delete_hnd(getfield(h, flds{ii}));
  end
elseif ~isempty(h),
  delete(h);
end
