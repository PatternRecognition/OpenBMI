function nogui_start_feedback(setup, varargin)

global VP_CODE

%opt= propertylist2struct(varargin{:});
%opt= set_defaults(opt, ...
%                  'impedances', 1);

if setup.general.save,
  if setup.savemode,
    warning('recording is still running: will stop it');
    bvr_stoprecording;
    pause(1)
  end

  filebase = setup.general.savestring;
  filename= bvr_startrecording([filebase VP_CODE], varargin{:});
  setup.savemode = true;
  pause(1)
end

nogui_send_setup(setup, 'control_player1','bbci.status=''play'';');
nogui_send_setup(setup, 'graphic_player1','feedback_opt.status=''play'';');
