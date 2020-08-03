function bbci_owner(subdir)
%BBCI_OWNER - Changes file permissions to group bbci
%
%Synopsis:
% bbci_owner(SUBDIR)
%
%Changes the permissions only to user and group read/write/executable
%and the group ownership to bbci. Operates recursively.
%
%If called without arguments, the function operates on the directory
%EEG_RAW_DIR (global variable). This will produce a lot of warnings
%that can be ignored.

if nargin==0,
  global EEG_RAW_DIR
  subdir= EEG_RAW_DIR;
end

if ~strncmp(subdir, '/home/data', 10),
  warning('not in FIRST network: cannot change group');
  return
end

cmd= sprintf('chmod -R a-rwx,ug+rwX %s', subdir);
unix_cmd(cmd, 'could not change permissions');
cmd= sprintf('chown -R :bbci %s', subdir);
unix_cmd(cmd, 'could not change ownership');
