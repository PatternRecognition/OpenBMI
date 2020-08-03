function check_sampling(divTr, divTe)
% checks sampling partitions as returned from sample_* functions
% for consistency.

if isempty(divTr),
  return;
end

if ~iscell(divTr) | ~iscell(divTe),
  error('sampling must be specified as cell arrays');
end

if ~iscell(divTr{1}) | ~iscell(divTe{1}),
  error('sampling must be specified as cell arrays OF cell arrays');
end

if length(divTr)~=length(divTe) | ...
      ~isequal(apply_cellwise(divTr, 'length'), ...
               apply_cellwise(divTe, 'length')),
  error('divTr and divTe mismatch in length');
end

for ii= 1:length(divTe),
  test_samples= [divTe{ii}{:}];
  if length(test_samples)~=length(unique(test_samples)),
    msg= sprintf('index sets of divTe{%d} are not disjoint', ii);
    error(msg);
  end
end
