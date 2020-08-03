function bbci_out= bbci_apply_cleanup(bbci)
% Get rid of the fields of the old-style classifier

bbci_out= copy_struct(bbci, 'source','marker','signal','feature','classifier', ...
                             'control','feedback','log','adaptation','quit_condition');
if ~isstruct(bbci_out.log),
  bbci_out= rmfield(bbci_out, 'log');
end
