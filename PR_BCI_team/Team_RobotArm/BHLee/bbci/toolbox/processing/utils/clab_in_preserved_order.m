function clab= clab_in_preserved_order(given_clab, selected_clab)

if isstruct(given_clab),
  if ~isfield(given_clab, 'clab');
    error('field ''clab'' in first argument missing');
  end
  given_clab= given_clab.clab;
end

selected= ismember(given_clab, selected_clab);
clab= given_clab(selected);
