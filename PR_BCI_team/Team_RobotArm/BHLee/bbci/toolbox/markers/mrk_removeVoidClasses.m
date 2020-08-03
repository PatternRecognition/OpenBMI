function [mrk, ev]= mrk_removeVoidClasses(mrk)
%mrk= mrk_selectEvents(mrk)
%
% requires a field 'y' containing the class labels in mrk.

if ~isfield(mrk, 'y'),
  error('class-label (mrk.y) required');
end

nonvoidClasses= find(any(mrk.y,2));
if length(nonvoidClasses)<size(mrk.y,1),
  msg= sprintf('void classes removed, %d classes remaining', ...
                  length(nonvoidClasses));
  bbci_warning(msg, 'mrk', mfilename);
  mrk.y= mrk.y(nonvoidClasses,:);
  if isfield(mrk, 'className'),
    mrk.className= {mrk.className{nonvoidClasses}};
  end
end
