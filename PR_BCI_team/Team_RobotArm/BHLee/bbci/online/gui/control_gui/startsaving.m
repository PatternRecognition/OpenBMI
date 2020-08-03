function startsaving(fig)

% global TODAY_DIR

setup = control_gui_queue(fig,'get_setup');

filebase = setup.general.savestring;

% if ~isabsolutepath(filebase),
%   filebase= fullfile(TODAY_DIR, filebase);
% end
fprintf('** Prepare recording to file %s*.\n', filebase);


ax = control_gui_queue(fig,'get_general_ax');
set(ax.interrupt, 'Enable','on');

set(ax.savefile,'Visible','off');
set(ax.savestring,'Visible','off');
set(ax.stop_marker,'Visible','off');
set(ax.stop_markertext,'Visible','off');
set(ax.interrupt,'Visible','on');
set(ax.interrupt,'Enable','off');
set(ax.interrupt,'String',sprintf('Interrupt saving to %s',filebase));
fprintf('Send pause to feedback\n');
playing_game(fig,'all','pause');
pause(1);

%flag = interrupt_timer(fig, setup.general.stopmarker);
flag = 1;

if flag,
  global VP_CODE
  filename= bvr_startrecording([filebase VP_CODE]);
  fprintf('Start recording to file %s.\n', filename);
  setup.savemode = true;
  set(ax.interrupt,'Enable','on');
  control_gui_queue(fig,'set_setup',setup);
else
  fprintf('Recording already started.\n');
end
pause(0.5);

fprintf('Initialize feedback\n');
