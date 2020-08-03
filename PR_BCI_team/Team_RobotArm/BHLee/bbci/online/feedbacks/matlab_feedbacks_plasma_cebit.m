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



feedback_opt = {};
run = true;

fig = gcf;

feedback_opt.reset = 1;
feedback_opt = set_defaults(feedback_opt,'parPort',1,'client_player',1);

old_control_port = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here comes the actual loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while run
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Initially: wait until the information about the feedback has arrived.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ~exist('control_port','var') | isempty(control_port)
        control_port = 12470;
    end
    
    if old_control_port~=control_port
        control_communication = get_data_udp(control_port);
        old_control_port = control_port;
    end
    
    while isempty(feedback_opt) | ~isstruct(feedback_opt) | ~isfield(feedback_opt,'type') 
        get_setup;
        if run==0
            return;
        end
        pause(0.1);
    end
    
    % if no information about server or client is given, this program runs
    % a monolyt
    
    feedback_opt = set_defaults(feedback_opt,'client',false,'graphic_master',false);
    
    send_data_udp(get_hostname,12345);
    
    
    run = true;
    %global cebit BCI_DIR
    %if isempty(cebit), cebit=0;end
    
    %open the control port
    figure(fig);
    set(fig,'MenuBar','none');
    
    loop = true;
    hilfstyp = 0;
    
    switch feedback_opt.type
        case 'feedback_hexawrite_init'
            handle = feval('feedback_hexa_init',fig,feedback_opt); % all graphic handles
            feedback_opt.client_ports = 12450+feedback_opt.client_player;
            
            hilfstyp = feedback_opt.client_player;
        case 'feedback_brainpong_client'
            global werbung werbung_opt BCI_DIR
            werbung = 1;
            werbung_opt.position = [320,42,640,42];
            werbung_opt.pictures = struct('image',strcat([DATA_DIR 'images/cebit_brainpong_logos'], {'1','2'}, '.png'),'position',{[0.2 0.8 0.6 0.2],[0.2 0 0.6 0.2]});
%            feedback_opt.position = [0 0 1280 768];
            handle = feval('feedback_brainpong_client',fig,feedback_opt); % all graphic handles
            feedback_opt.client_ports = 12450;
            hilfstyp = 3;
            
    end        
    if isstruct(handle)
        handle= fb_handleStruct2Vector(handle);
    end
    
    
    if isfield(feedback_opt,'client_ports')    
        client_comm = get_data_udp(feedback_opt.client_ports);
    else
        client_comm = get_data_udp(12450);
    end
    
    
    
    drawnow;
    
    loop = true;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % here comes the actual loop.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while loop
        
        a= get_data_udp(client_comm,0.1);
        
        if ~isempty(a)    % eval a
            try  % Todo: Catch Ctr-C for stopping feedback
                a = eval(char(a));
                for i = 1:length(a)
                    if ~isnan(handle(a{i}{1}))
                        set(handle(a{i}{1}),a{i}{2:end});
                    end
                end
            end
            
        end
        drawnow; % hopefully drawnow does nothing if nothing is to update
        
        % Any new setup variables?
        get_setup;
        if strcmp('loop=false;run=1;',new_setup);
            loop= true;
        end
        send_data_udp(hilfstyp);
        
    end
    
    
    send_data_udp;
    get_data_udp(client_comm);
    
    if old_control_port~=control_port | ~run
        get_data_udp(control_communication); % close connection to gui
    end
    
end

if ~isempty(fig);
    close(fig);
end





