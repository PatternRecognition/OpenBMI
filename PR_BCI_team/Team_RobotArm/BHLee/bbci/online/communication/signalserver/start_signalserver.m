function start_signalserver(configFile)
% start_signalserver
%
% SYNOPSIS
%   start_signalserver(configFile)
% 
% ARGUMENTS
%   configFile: The configuration file for the signal server. You should
%               give an absoulute path here. Or a path relative to the
%               signal server directory.
%
% AUTHOR
%    Max Sagebaum 
%
%    2010/08/30 Max Sagebaum
%                    - file created
  currentFolder = pwd();
  signalServerFolder = strrep(which('signalserver.exe'),'signalserver.exe','');
  
  %start the 
  cd(signalServerFolder);
  system(['signalServer.exe -o -f ' configFile ' &']);
  cd(currentFolder);