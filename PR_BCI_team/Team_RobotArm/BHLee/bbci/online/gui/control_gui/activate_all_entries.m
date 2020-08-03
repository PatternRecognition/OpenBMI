function activate_all_entries(fig);
% ACTIVATE_ALL_ENTRIES ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% activate all entries of the gui
%
% usage:
%      activate_all_entries(fig);
%
% input:
%     fig   the handle of the gui
%
% Guido Dornhege
% $Id: activate_all_entries.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $


% get setup and all entries
setup = control_gui_queue(fig,'get_setup');

ax = control_gui_queue(fig,'get_general_ax');

setup.general.player = get(ax.player,'Value');
setup.general.graphic = get(ax.graphic,'Value');

ax = control_gui_queue(fig,'get_control_player1_ax');
setup.control_player1.machine = get(ax.machine,'String');
setup.control_player1.port = str2num(get(ax.port,'String'));

fi = ax.fields;
if prod(size(fi))>1
  for i = 1:size(fi,1);
    na = get(fi(i,1),'String');
    if na(1)=='.'
      na = ['bbci', na];
    end
    va = get(fi(i,2),'String');
    try
      eval(sprintf('setup.control_player1.%s = %s;',na,va));
    catch
      set(fig,'Visible','off');
      drawnow;
      message_box('Check the entries',1);
      set(fig,'Visible','on');
      return;
    end
    
  end
end

if setup.general.player>1
  ax = control_gui_queue(fig,'get_control_player2_ax');
  setup.control_player2.machine = get(ax.machine,'String');
  setup.control_player2.port = str2num(get(ax.port,'String'));
  
  fi = ax.fields;
  if prod(size(fi))>1
    for i = 1:size(fi,1);
      na = get(fi(i,1),'String');
      if na(1)=='.'
        na = ['bbci', na];
      end
      va = get(fi(i,2),'String');
      try
        eval(sprintf('setup.control_player2.%s = %s;',na,va));
      catch
        set(fig,'Visible','off');
        drawnow;
        message_box('Check the entries',1);
        set(fig,'Visible','on');
        return;
      end

        
    end
  end
  
end

if setup.general.graphic
  ax = control_gui_queue(fig,'get_graphic_player1_ax');
  setup.graphic_player1.machine = get(ax.machine,'String');
  setup.graphic_player1.port = str2num(get(ax.port,'String'));
  setup.graphic_player1.fb_port = str2num(get(ax.fb_port,'String'));
  
  fi = ax.fields;
  if prod(size(fi))>1
  
    for i = 1:size(fi,1);
      na = get(fi(i,1),'String');
      if na(1)=='.'
        na = ['feedback_opt', na];
      end
      va = get(fi(i,2),'String');
      try
        eval(sprintf('setup.graphic_player1.%s = %s;',na,va));
      catch
        set(fig,'Visible','off');
        drawnow;
        message_box('Check the entries',1);
        set(fig,'Visible','on');
        return;
      end
        
    end
  end
  if setup.general.player>1
    ax = control_gui_queue(fig,'get_graphic_player2_ax');
    setup.graphic_player2.machine = get(ax.machine,'String');
    setup.graphic_player2.port = str2num(get(ax.port,'String'));
    setup.graphic_player2.fb_port = str2num(get(ax.fb_port,'String'));

    fi = ax.fields;
    if prod(size(fi))>1
      for i = 1:size(fi,1);
        na = get(fi(i,1),'String');
        if na(1)=='.'
          na = ['feedback_opt', na];
        end
        va = get(fi(i,2),'String');
        try
          eval(sprintf('setup.graphic_player2.%s = %s;',na,va));
        catch
          set(fig,'Visible','off');
          drawnow;
          message_box('Check the entries',1);
          set(fig,'Visible','on');
          return;
        end
          
      end
    end
  end
end

  

% save

control_gui_queue(fig,'set_setup',setup);
