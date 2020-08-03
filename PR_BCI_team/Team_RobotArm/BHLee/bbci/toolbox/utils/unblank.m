function strs= unblank(strs)

if ischar(strs),
  strs= fliplr(deblank(fliplr(deblank(strs))));
  return;
end

strs= deblank(strs);
strs= apply_cellwise(strs, 'fliplr');
strs= deblank(strs);
strs= apply_cellwise(strs, 'fliplr');
