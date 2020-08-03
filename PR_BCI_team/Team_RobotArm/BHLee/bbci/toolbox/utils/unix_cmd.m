function out= unix_cmd(cmd, action)
%UNIX_CMD - Run a unix command and check for errors
%
%Synopsis:
% unix_cmd(CMD, <ACTION>)
%
%Arguments:
% CMD:    String, the unix command to be executed
% ACTION: String, printed in case of an error


if nargin<2,
  action= '';
end

[stat,out]= unix(cmd);
if stat~=0,
  error(sprintf('error %s (%s -> %s)', action, cmd, out));
end

if nargout==0,
  clear out
end
