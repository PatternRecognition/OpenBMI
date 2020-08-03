function mrk= mrk_addIndexedField(mrk, flds)

if ~iscell(flds),
  if ~ischar(flds),
    error('2nd argument must be a char, or a cell array of such.');
  end
  flds= {flds};
end
ic= apply_cellwise(flds, 'ischar');
if ~all([ic{:}]),
  error('all cell elements of the 2nd argument must be chars');
end
for ff= 1:length(flds),
  if ~isfield(mrk, flds{ff}),
    warning(sprintf('%s is not (yet?) a field in the marker structure', ...
                    flds{ff}));
  end
end

if isfield(mrk, 'indexedByEpochs'),
  mrk.indexedByEpochs= unique_unsort(cat(2, mrk.indexedByEpochs, flds));
else
  mrk.indexedByEpochs= flds;
end
