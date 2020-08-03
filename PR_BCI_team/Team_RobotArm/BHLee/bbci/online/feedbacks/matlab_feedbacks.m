% 
%
% used as a gateway for three different types of feedback settings:
% monolith: only one feedback module, no further clients
% master: master in a master/client setting
% client: client in a master/client setting
%
% master/client setting is needed when several players need to be served with a 
% common application logic (e.g. for brainpong).
% monolith is the default solution when only one player is present (e.g.
% for spelling)
%
% 11.Jan 2006 Guido, Michael

global general_port_fields 

if ~exist('player','var') | isempty(player)
  player = 1;
end

feedback_opt = {};
run = true;

fig = [];

feedback_opt.reset = 1;
old_control_port = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here comes the actual loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while run
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initially: wait until the information about the feedback has arrived.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~exist('control_port','var') | isempty(control_port)
      if ~isempty(general_port_fields) & isfield(general_port_fields,'graphic') & ~isempty(general_port_fields(min(length(general_port_fields),player)).graphic)
        control_port = general_port_fields(player).graphic{2};
      else
        control_port = 12470;
      end
    end
    
    if old_control_port~=control_port
        control_communication = get_data_udp(control_port);
        old_control_port = control_port;
    end
    
    while isempty(feedback_opt) | ~isstruct(feedback_opt) | ~isfield(feedback_opt,'type') 
        pause(0.1);
        get_setup;
        if run==0
            return;
        end
    end
    feedback_opt = set_defaults(feedback_opt,'parPort',1,'log',1);
    if isfield(feedback_opt,'parPort') & feedback_opt.parPort==1
        do_set(255);
    end
   
    % if no information about server or client is given, this program runs
    % a monolyt
    
    feedback_opt = set_defaults(feedback_opt,'client',false,'graphic_master',false);
    
    if feedback_opt.graphic_master
        if ~isempty(fig); close(fig); fig = [];end
        [run,feedback_opt] = matlab_feedbacks_master(feedback_opt,control_communication);
    elseif feedback_opt.client
        if isempty(fig); fig = figure; end
        [run,feedback_opt] = matlab_feedbacks_client(feedback_opt,control_communication,fig);
    else
        if isempty(fig); fig = figure; end
        [run,feedback_opt] = matlab_feedbacks_monolith(feedback_opt,control_communication,fig);
    end        
    
    if old_control_port~=control_port | ~run
        get_data_udp(control_communication); % close connection to gui
    end
   
    
end

if ~isempty(fig);
    close(fig);
end



  
  
