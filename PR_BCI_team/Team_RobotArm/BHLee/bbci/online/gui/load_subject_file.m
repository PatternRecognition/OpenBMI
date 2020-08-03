function bbci = load_subject_file(fi,pa);
% load a subject file fi in path pa
% see gui_file_setup and training_bbci_bet
% 
% TODO: extended documentation by Schwaighase
% Guido Dornhege, 07/03/2005
% $Id: load_subject_file.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $


bbci.train_file = {};
bbci.player = 1;
bbci.setup = [];
bbci.classDef = {};
bbci.feedback = [];

if isempty(fi)
  return;
end

d = cd;

try 
  cd(pa);
  if strcmp(fi(end-1:end),'.m')
    fi = fi(1:end-2);
  end
  eval(fi);
catch
  bbci = {};
end


cd(d);

if iscell(bbci) & ~isempty(bbci)
  global choose_one_opt_status
  % function with different options
  % OPENING THE FIGURE
  fig = figure;   
  scs = get(0,'ScreenSize');
  set(fig,'Position',[scs(3)/2 scs(4)/2 scs(3)/2-1 scs(4)/2-1]);
  set(fig,'MenuBar','none');
  set(fig,'NumberTitle','off');
  set(fig,'Name','choose one opportunity');
  set(fig,'Units','Normalized');
  set(fig,'DeleteFcn','global choose_one_opt_status;choose_one_opt_status = ''can'';');
  
  % THE CANCEL BUTTON
  can = uicontrol('Units','Normalized','Position',[0.12 0.02 0.25 0.2]);
  set(can,'Style','pushbutton');
  set(can,'String','cancel');
  set(can,'FontUnits','Normalized');
  set(can,'FontSize',0.6);
  set(can,'Callback','global choose_one_opt_status;choose_one_opt_status = ''can'';');
  set(can,'Tooltipstring','Do not choose one');

  ok = uicontrol('Units','Normalized','Position',[0.62 0.02 0.25 0.2]);
  set(ok,'Style','pushbutton');
  set(ok,'String','ok');
  set(ok,'FontUnits','Normalized');
  set(ok,'FontSize',0.6);
  set(ok,'Callback','global choose_one_opt_status;choose_one_opt_status = ''ok'';');
  set(ok,'Tooltipstring','Choose the marked one');

  ty = uicontrol('Units','Normalized','Position',[0.1,0.25,0.8,0.7]);
  set(ty,'Style','listbox');
  set(ty,'String',bbci);
  set(ty,'FontUnits','Normalized');
  set(ty,'FontSize',0.15);
  set(ty,'Value',1);
  set(ty,'Tooltipstring','choose one opportunity');
  set(ty,'Max',1,'Min',1);

  choose_one_opt_status = '';
  
  while true;
    drawnow;
    pause(0.1);
    if ~isempty(choose_one_opt_status)
      switch choose_one_opt_status
       case 'can'
        bbci = {};
       case 'ok'
        bbci = bbci{get(ty,'Value')};
        try
          eval(sprintf('%s(''%s'');',fi,bbci));
        catch
          bbci = {};
        end
      end
      delete(fig);
      break;
    end
  end
end

if ~isempty(bbci)
  bbci = set_defaults(bbci,'train_file',{{}},'player',1,'setup',[],'classDef',{{}},'feedback',[]);
end
