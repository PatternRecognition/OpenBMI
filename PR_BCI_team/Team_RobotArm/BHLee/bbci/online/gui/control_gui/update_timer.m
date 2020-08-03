function update_timer(tt,nval,typ,player)

global tim 
persistent control_player1 control_player2 graphic_player1 graphic_player2
persistent handle_control_player1 handle_control_player2 handle_graphic_player1 handle_graphic_player2

if nargin==0
  if ~isempty(tim);
    stop(tim);
    delete(tim);
  end
  tim = [];
  try;get_data_udp(handle_control_player1);end
  try;get_data_udp(handle_control_player2);end
  try;get_data_udp(handle_graphic_player1);end
  try;get_data_udp(handle_graphic_player2);end
  control_player1 = [];
  control_player2 = [];
  graphic_player1 = [];
  graphic_player2 = [];
  handle_control_player1 = [];
  handle_control_player2 = [];
  handle_graphic_player1 = [];
  handle_graphic_player2 = [];
  return
end

if nargin==1
  setup = control_gui_queue(tt,'get_setup');
  for typ = {'control','graphic'}
    for player = 1:2
      eval(sprintf('hand = handle_%s_player%d;',typ{1},player));
      if ~isempty(hand)
         val = get_data_udp(hand,0);
        if ~isempty(val)
          ax = control_gui_queue(tt,sprintf('get_%s_player%d_ax',typ{1},player));
          eval(sprintf('setu = setup.%s_player%d;',typ{1},player));        
          val = char(val);
          eval(sprintf('val = %s;',val));
          redraw = false;
          for ii = 1:2:length(val)
            fie = find(strcmp(setu.fields,val{ii}));
            if isempty(fie)
              setu.fields = {setu.fields{:},val{ii}};
              setu.fields_help = {setu.fields_help{:},[]};
              fie = length(setu.fields);  
              redraw = true;
            end
            eval(sprintf('setu.%s = val{ii+1};',val{ii}));
            if ~redraw
              set(ax.fields(fie,2),'String',get_text_string(val{ii+1}));
            end
          end
          eval(sprintf('setup.%s_player%d = setu;',typ{1},player));
          control_gui_queue(tt,'set_setup',setup);
          if redraw
            switch typ{1}
              case 'control'
                plot_control_gui(tt,player);
              case 'graphic'
                plot_graphic_gui(tt,player);
            end
          end
        end
      end
    end
  end

  return
end

setup=control_gui_queue(tt,'get_setup');

%eval(sprintf('oldv = setup.%s_%s.update_port;',typ,player));


if ~isnumeric(nval) | length(nval)>1 
  return;
end

eval(sprintf('infos = %s_%s;',typ,player));

if ~isempty(infos)
  % port schliessen
  try 
    get_data_udp(sprintf('handle_%s_%s',typ,player));
  end
  eval(sprintf('handle_%s_%s = [];',typ,player));
end

if ~isempty(nval)
  % port öffnen
  try
    eval(sprintf('handle_%s_%s = get_data_udp(%d);',typ,player,nval));
  catch
    nval = [];  
  end
end

eval(sprintf('%s_%s=nval;',typ,player));

if isempty(tim) & (~isempty(control_player1) | ~isempty(control_player2) | ~isempty(graphic_player1) | ~isempty(graphic_player2))
  % start timer
  tim = timer('StartDelay',1,'ExecutionMode','fixedSpacing','Period',1,'TimerFcn',sprintf('update_timer(%d);',tt),'BusyMode','queue');
  start(tim);
end

if ~isempty(tim) & isempty(control_player1) & isempty(control_player2) & isempty(graphic_player1) & isempty(graphic_player2)
  try
    stop(tim);
  end
  delete(tim);
  tim = [];
end

  

eval(sprintf('setup.%s_%s.update_port = nval;',typ,player));

control_gui_queue(tt,'set_setup',setup);
