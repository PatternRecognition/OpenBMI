function signalserver_impedance(configFile)
% start impedance measurement for signalserver
% Johannes
global BCI_DIR

if nargin < 1
    configFile = 'server_config.xml';
    % a standard 16channel setup
end

currentFolder = pwd();

% signalServerFolder = strrep(which('signalserver.exe'),'signalserver.exe','');
% cd(signalServerFolder);  

%start the
cd([BCI_DIR 'signal-aquisition\signalserver_impedance']);
system(['impedance.exe ' [BCI_DIR 'online\communication\signalserver\' configFile] ' &']);
cd(currentFolder);