function im_lucky(fig, varargin);
% IM_LUCKY ONLY FOR INTERNAL USE OF MATLAB_CONTROL_GUI
%
% finds and loads the most probable classifier setup

global TODAY_DIR EEG_RAW_DIR

opt = propertylist2struct(varargin{:});

set(fig,'Visible','off');
drawnow;
setup = control_gui_queue(fig,'get_setup');

for player = 1:2
  if setup.general.player<player
    continue;
  end
  if player==1 & ~setup.general.active1
    continue;
  end
  if player==2 & ~setup.general.active2
    continue;
  end

  eval(sprintf('machine = setup.control_player%d.machine;',player));
  eval(sprintf('port = setup.control_player%d.port;',player));

  if isempty(TODAY_DIR),

    warning('global variable TODAY_DIR must be set');

    dire = EEG_RAW_DIR;
    da = datevec(now);
    da = sprintf('_%02d_%02d_%02d',mod(da(1:3),100));

    directories = get_directory_info([], dire, player, machine, port);

    if isnumeric(directories)
      continue;
    end

    di = {};

    for i = 1:length(directories)
      if ~isempty(strfind(directories{i},da))
        di = {di{:},directories{i}};
      end
    end

    if isempty(di)
      error('No directory contains the date in its name')
    end
  else
    % changed by Claudia: before was di = TODAY_DIR
    di{1} = TODAY_DIR;
  end
  fili = {}; classi = {};

  for i = 1:length(di)
    % changed by Claudia. Before was:
    %     [dum,files,players,classes] = ...
    %       get_directory_info([],[dire di{i}],player,machine,port);
    % but dire doesn't exist if today_dir isempty. Moreover di contains
    % already the path to search
    [dum,files,players,classes] = ...
      get_directory_info([],di{i},player,machine,port);
    if ~isnumeric(dum) && length(files)>0
      ind = find(players==player);
      for j = 1:length(ind)
        fili = {fili{:},[di{i} files{ind(j)}]};
      end
      classi = {classi{:},classes{ind}};
    end
  end


  if length(fili) == 0
    error('No files mat have been found')
  end

  if length(fili)>1
    if isfield(opt,'classifier')
      if isnumeric(opt.classifier)
        if length(fili) >= opt.classifier
         fili = fili(opt.classifier);
         classes = classes(opt.classifier);
        else
         error(['Maximum number of classifier available is ' int2str(length(fili))]);
        end
      elseif isequal(opt.classifier,'auto')
        disp('Select the last classifier')
        fili = fili(end);
        classes = classes(end);
      else
        disp('No valid parameter for the option classifier: please choose one classifier from the list');
      end
    else
      [fili classes] = ask_for_fili(fig,fili,classi,player);
    end
  end

  if length(fili)>0
    ax = control_gui_queue(fig,'get_general_ax');

    eval(sprintf('h = ax.setup_list%d;',player));
    
    for i = 1:length(fili)
      filis{i} = sprintf('%s (%s)',fili{i},classes{i});
      set(h,'String',filis,'Value',1);
    end
    
    eval(sprintf('setup.general.setup_list%d = fili;',player));
    control_gui_queue(fig,'set_setup',setup);

  end
end

idx = findstr(classes{1},'/');
setup.graphic_player1.feedback_opt.classes{1} = classes{1}(1:idx-1);
setup.graphic_player1.feedback_opt.classes{2} = classes{1}(idx+1:end);
control_gui_queue(fig,'set_setup',setup);

set(fig,'Visible','on');drawnow;




function [fili, classes] = ask_for_fili(fig,fili,classi,player);

global nice_gui_font ask_for_fili

fi = figure;
set(fi,'MenuBar','None','CloseRequestFcn','global ask_for_fili; ask_for_fili=0;','Position',get(fig,'Position'));
set(fi,'Color',[0.9,0.9,0.9]);
set(fi,'NumberTitle','off')
set(fi,'Name',['CHOOSE A FILE FOR PLAYER ' int2str(player)]);


canc = uicontrol('Style','pushbutton','units','normalized','position',[0.05 0 0.4 0.1],'String','Cancel');
set(canc,'Tooltipstring','do not choose one');
set(canc,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(canc,'Callback','global ask_for_fili; ask_for_fili=0;');

ok = uicontrol('Style','pushbutton','units','normalized','position',[0.55 0 0.4 0.1],'String','Ok');
set(ok,'Tooltipstring','choose the selected one');
set(ok,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.5);
set(ok,'Callback','global ask_for_fili; ask_for_fili=1;');

filis = cell(1,length(fili));
for i = 1:length(fili)
  filis{i} = sprintf('%s (%s)',fili{i},classi{i});
end

lis = uicontrol('Style','listbox','units','normalized','position',[0.1 0.2 0.8 0.7],'String',filis,'Value',1);
set(lis,'Tooltipstring','choose the files');
set(lis,'FontUnits','normalized','FontName',nice_gui_font,'FontSize',0.05);
set(lis,'Min',0,'Max',length(fili));


ask_for_fili = [];

while isempty(ask_for_fili)
  pause(0.1);
end

if ask_for_fili
  fili = fili(get(lis,'Value'));
  classes = classi(get(lis,'Value'));
else
  fili = {};
end

if length(fili) > 1
  warning('Instability will be caused chosen more then 1 classifier at the same time. Classes will be set from the first classifier');
end

delete(fi);




return;
