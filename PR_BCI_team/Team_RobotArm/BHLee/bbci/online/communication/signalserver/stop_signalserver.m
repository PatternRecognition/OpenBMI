function stop_signalserver()
% stop_signalserver
%
% SYNOPSIS
%   stop_signalserver()
% 
% AUTHOR
%    Max Sagebaum 
%
%    2010/08/30 Max Sagebaum
%                    - file created
  dos('taskkill /F /IM signalserver.exe');
