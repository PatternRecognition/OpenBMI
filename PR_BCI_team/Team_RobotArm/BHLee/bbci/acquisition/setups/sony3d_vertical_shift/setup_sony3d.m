global tcp_conn;

addpath([BCI_DIR 'import/tcp_udp_ip']);

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

VP_SCREEN = [0 0 1280 1024]

try
  bvr_checkparport('type','S');
catch
  error('Check amplifiers (all switched on?) and trigger cables.');
end

global TODAY_DIR
acq_makeDataFolder;
mkdir([TODAY_DIR 'data']);


pnet('closeall');
tcp_conn = pnet('tcpconnect', 'sony3d.ml.tu-berlin.de', 12345);
if tcp_conn<0
  error('Couldn''t connect to server.');
end
pnet(tcp_conn, 'setwritetimeout', 1)
pnet(tcp_conn, 'printf', 'msg waiting_for_experiment\n');

fprintf('Press ENTER when display shows message\n');
pause;
