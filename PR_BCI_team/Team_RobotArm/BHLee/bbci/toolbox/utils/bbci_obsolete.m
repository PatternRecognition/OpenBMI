function bbci_obsolete(oldFile, newFile)

if nargin==1 | isempty(newFile),
  msg= sprintf('<%s> is obsolete.', oldFile);
else
  msg= sprintf('<%s> is obsolete: use <%s> in future.', oldFile, newFile);
end
bbci_warning(msg, 'obsolete', oldFile);
