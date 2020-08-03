pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);
%send_xmlcmd_udp('init', '127.0.0.1', 12345);

pyff('quit');

pyff('init','ZweiHand'); pause(.5)
pyff('set', 'fullscreen',false);
pyff('set', 'sound',true);
pyff('set', 'log_dir', TMP_DIR);
pyff('set', 'cursor_color', 'mediumseagreen');
pyff('setint', 'running_time', 120);
geometry= [0 0 800 600];
pyff('setint','geometry',geometry);
pyff('play');

