global tcp_conn;

addpath([BCI_DIR 'import/tcp_udp_ip']);

if isempty(VP_CODE),
  warning('VP_CODE undefined - assuming fresh subject');
end

VP_SCREEN = [0 0 1280 1024]

if record
  try
    bvr_checkparport('type','S');
  catch
    error('Check amplifiers (all switched on?) and trigger cables.');
  end
end

global TODAY_DIR
acq_makeDataFolder('multiple_folder',1)
%mkdir([TODAY_DIR 'data']);

pnet('closeall');
tcp_conn = pnet('tcpconnect', 'sony3d.ml.tu-berlin.de', 12345);
if tcp_conn<0
  error('Couldn''t connect to server.');
end
pnet(tcp_conn, 'setwritetimeout', 1)
pnet(tcp_conn, 'printf', 'freq %d %d %d\n',640,480,2*99);
fprintf('Press ENTER when CRT has entered fullscreen mode\n');
pause;

