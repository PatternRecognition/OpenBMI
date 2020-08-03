function add_setup(fig,player, varargin);
% ADD_SETUP ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% adds a setup in the gui
%
% usage:
%    add_setup(fig,player);
% 
% input:
%    fig     the handle of the gui
%    player  player number
%
% Guido Dornhege
% $Id: add_setup.m,v 1.2 2006/05/22 12:23:19 neuro_cvs Exp $
global REMOTE_RAW_DIR
if isempty(REMOTE_RAW_DIR)
  REMOTE_RAW_DIR = '/home/neuro/data/BCI/bbciRaw/';
end

opt = propertylist2struct(varargin{:});

set(fig,'Visible','off');drawnow;

ax = control_gui_queue(fig,'get_general_ax');
setup = control_gui_queue(fig,'get_setup');

eval(sprintf('h = ax.setup_list%d;',player));

str = get(h,'String');
va = get(h,'Value');


if va==0 || va>length(str)
  dire = REMOTE_RAW_DIR;
else
  c = strfind(str{va},'/');
  if isempty(c)
    dire = REMOTE_RAW_DIR;
  else
    dire = str{va}(1:c(end));
  end
end

eval(sprintf('machine = setup.control_player%d.machine;',player));
eval(sprintf('port = setup.control_player%d.port;',player));

if isfield(opt,'classifier')
  if isabsolutepath(opt.classifier)
    strnew = opt.classifier;
  elseif isequal(opt.classifier, 'auto')
    [strnew classes] = gui_add_setup(fig,dire,player,machine,port,'auto');
  else
    strnew = [dire opt.classifier];
  end
else
  [strnew, classes] = gui_add_setup(fig,dire,player,machine,port);
end
if ~exist('classes', 'var'),
  [dire, cfyname]= fileparts(strnew);
  [a,cfyfiles,c,classes] = get_directory_info(fig,dire,player,machine,port);
  idx= strmatch(cfyname, cfyfiles);
  classes= classes{idx};
end

set(fig,'Visible','on');

if ~isempty(strnew)  

  if va<=length(str)
    va = va+1;
  end
  
  str = {str{1:va-1},strnew,str{va:end}};
    
  set(h,'String',[str{1} ' (' classes ')'],'Value',va);
  
  eval(sprintf('setup.general.setup_list%d = str;',player));
  control_gui_queue(fig,'set_setup',setup);
end

idx = findstr(classes,'/');
setup.graphic_player1.feedback_opt.classes{1} = classes(1:idx-1);
setup.graphic_player1.feedback_opt.classes{2} = classes(idx+1:end);
control_gui_queue(fig,'set_setup',setup);

