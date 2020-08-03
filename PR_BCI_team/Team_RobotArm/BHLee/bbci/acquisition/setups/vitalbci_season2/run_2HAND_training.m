%session_name= 'vitalbci_season2';
%acqFolder = [BCI_DIR 'acquisition/setups/' session_name '/'];
%ZHAND_file= [acqFolder 'ZHAND'];

pyff('startup','a',[BCI_DIR 'python\pyff\src\Feedbacks'], 'bvplugin', 0);
fprintf('Starting Pyff...\n'); pause(10);

pyff('init','ZweiHand'); pause(3.5)
%pyff('load_settings', ZHAND_file);
pyff('set', 'fullscreen',false);
pyff('set', 'sound',true);
pyff('set', 'log_dir', TODAY_DIR);
pyff('setint', 'running_time', 22*60);
pyff('set', 'cursor_color', 'mediumseagreen');
PrimaryScreenSize= get(0, 'ScreenSize');
%%n case it doesn't work (for psylab): 
%PrimaryScreenSize=[0 0 2560 1600];
screen_geometry= VP_SCREEN;
screen_geometry(2)= PrimaryScreenSize(4) - VP_SCREEN(4);
game_size= [1000 660];
geometry= screen_geometry([1 2]) + ...
  (screen_geometry([3 4])-game_size)/2;
geometry([3 4])= game_size;

switch(1),
  case {1, 'Drehregler'},
    pyff('set', 'joystick_style','absolute');
    pyff('setint','x_axis_id',[0 1], 'y_axis_id',[0 0]);
    pyff('set', 'cursor_speed', 50000);
  case {2, 'Joystick'},
    pyff('set', 'joystick_style','relative');
    pyff('setint','x_axis_id',[0 1], 'y_axis_id',[0 0]);
    %pyff('set', 'movement_threshold',.5);
    pyff('set', 'cursor_speed', geometry(4));
  case {3, 'DrehreglerUnlimited'},
    pyff('set', 'joystick_style','button_push');
    pyff('setint','x_axis_id',[0 3 2], 'y_axis_id',[0 1 0]);
    pyff('set', 'cursor_speed', 2000);
    pyff('set', 'interval', 0.01);
end
pyff('setint','geometry',geometry);

stimutil_waitForInput('msg_next','to start 2HAND training.');
pyff('play');
%stimutil_waitForMarker('S255');
stimutil_waitForInput('msg_next','when 2HAND training has finished.');
pyff('quit');
fprintf('Close Pyff window.\n')
