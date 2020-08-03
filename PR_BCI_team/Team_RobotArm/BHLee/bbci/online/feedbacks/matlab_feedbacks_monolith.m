function [run,feedback_opt] = matlab_feedbacks_monolith(feedback_opt,control_communication,fig);
% 
% Realization of rather simple feedback modes with only one player or if
% players are completely independant (e.g. for spelling)
% 
% matlab_feedbacks_monolith is called by matlab_feedbacks
%
% For Cebit, an extra output is included that serves a seperate screen.
% This feature is treated like a master/client version and _should_
% actually not be in here!!
%
% 11.Jan 2006 Guido, Michael

global lost_packages


run = true;

%open the control port
if isfield(feedback_opt,'fb_port')    
    monolith_com = get_data_udp(feedback_opt.fb_port);
else
    monolith_com = get_data_udp(12489);
end

set(fig,'MenuBar','none');


% set some defaults, call the init etc.
feedback_opt.reset = 1;
loop = true;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% here comes the actual loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mar = -1;


while loop
    
 a= get_data_udp(monolith_com,0);
    
 % if packet was not empty, try to get all waiting packets
 if ~isempty(a)
     b = a;
     while ~isempty(b)
         a = b;
         % read header of packet
         lg_no = a(1); % number of log file
         bl_no = a(2); % block number
         timestamp = a(3);
         pa = a(4); % How many parallel port markers are needed?
         if feedback_opt.log
             % do some logging.
             do_set('counter',bl_no,timestamp,lg_no);
         end
         for i = 1:pa % send markers 
            do_set(a(i+4)); 
         end
         
         % get next packet
         b = get_data_udp(monolith_com,0); % when last packet is reached, b will be empty and a contains last valid packet
     end
     
      % Anything lost during the transmission?
         
     if bl_no>mar+1 & mar>=0
         fprintf('%i packages lost\n',bl_no-mar-1);
         lost_packages = bl_no-mar-1;
     else
         lost_packages = 0;
     end

     mar = bl_no; %remember old block number
     dat = a(pa+5:end); % actual control data
     dat = num2cell(dat);
     
     % Call the feedback visualization function.
     feedback_opt = feval(feedback_opt.type,fig,feedback_opt,dat{:});
 end
 
 % check if new options are available at control port
 get_setup;
 if ~isempty(new_setup)
     fprintf('%s\n',new_setup);
     if strcmp(new_setup,'loop=false;run=1;')
         loop = true;
     else
        do_set(235); % send marker: 
        feedback_opt.changed = 1;
    end 
 end

 if isempty(new_setup) & isempty(a) & rand>0.99
     pause(0.00001); % one is able to interrupt
 end
end
if isfield(feedback_opt,'parPort') & feedback_opt.parPort==1
  do_set(210);
end

do_set('exit',run);
get_data_udp(monolith_com); %close feedback port



