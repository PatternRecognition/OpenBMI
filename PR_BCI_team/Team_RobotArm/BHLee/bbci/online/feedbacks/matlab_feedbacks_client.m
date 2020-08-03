function [run,feedback_opt] = matlab_feedbacks_client(feedback_opt,control_communication,fig);
% 
% Realization of framework for multi-player feedback mode. 
% Client function receives commands of the form set... from master. 
% Client inits and then visualizes objects.
%
% matlab_feedbacks_client is called by matlab_feedbacks
%
% 11.Jan 2006 Guido, Michael

run = true;
%global cebit BCI_DIR
%if isempty(cebit), cebit=0;end

feedback_opt = set_defaults(feedback_opt,'client_port',12450);

%open the control port
client_comm = get_data_udp(feedback_opt.client_port);

figure(fig);
set(fig,'MenuBar','none');

loop = true;

handle = feval(feedback_opt.type,fig,feedback_opt); % all graphic handles
if isstruct(handle)
    handle= fb_handleStruct2Vector(handle);
end

feedback_opt = rmfield(feedback_opt,{'client','client_player'});    


drawnow;

loop = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here comes the actual loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while loop

    a= get_data_udp(client_comm,0.04);
    
if ~isempty(a)    % eval a
    a = eval(char(a));
    for i = 1:length(a)
      if ~isnan(handle(a{i}{1}))
        set(handle(a{i}{1}),a{i}{2:end});
      end
    end
end 
%    if cebit==1
%        set(handle(4),'Rotation',0);
%    end 
    drawnow; % hopefully drawnow does nothing if nothing is to update
        
    % Any new setup variables?
    get_setup;
    if strcmp(new_setup,'loop=false;run=1;')
        loop = true;
    end
end
  
  
get_data_udp(client_comm);