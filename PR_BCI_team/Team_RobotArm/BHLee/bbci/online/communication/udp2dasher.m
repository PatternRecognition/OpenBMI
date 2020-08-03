function udp2dasher(varargin)
% udp2dasher(<opt>)
% udp2dasher('hostname','Brainamp','input_port',xxx,...)
% sends any UDP signals in bbci_bet format to another port in dasher
% format.
%
% IN:   opt     - options struct, possible fields:
%       .hostname   - where to send the dasher UDP. <localhost>
%       .input_port - where to listen to bbci_bet-UDP. <12489>
%       .output_port- where to send the dasher UDP. <20320>
%       .x_range    - range to which the x-data are clipped. <[-1,1]>
%       .y_range    - range to which the y-data are clipped. <[-1,1]>
%       .x_label    - label which is expected by the dasher. <x>
%       .y_label    - label which is expected by the dasher. <y>
%       .fs_out     - frequency for sending the dasher UDP. <15>
%       .verbosity  - display sent packages <0>
%
% USAGE: udp2dasher('hostname','Brainamp','verbosity',2)
%       

% kraulem 11/05
if nargin<1
    opt=struct;
elseif nargin>1
    opt=propertylist2struct(varargin{:});
end 
opt = set_defaults(opt,'hostname','localhost',...
                    'input_port',12489,...
                    'output_port',20320,...
                    'x_range',[-1 1],...
                    'x_label','x',...
                    'y_range',[-1,1],...
                    'y_label','y',...
                    'fs_out',15,...
                    'verbosity',0);

% open the connections:
try
    get_udp(opt.hostname, opt.input_port);
catch
    get_udp('close');
    get_udp(opt.hostname, opt.input_port);
end  
send_udp_dasher(opt.hostname,opt.output_port,opt.verbosity);  

% do this loop until 10 seconds without data.
curr_data = 1;
while ~isempty(curr_data)
    data=get_udp(10);
    if isempty(data)
        % no packages available.
        curr_data = data;
        break;
    end
    while ~isempty(data)
        % empty the queue
        curr_data=data;    
        data=get_udp(.001);    
    end

    if length(curr_data)>5,
        curr_data(5) = max(min(curr_data(5),opt.x_range(2)),opt.x_range(1));
        curr_data(6) = max(min(curr_data(6),opt.y_range(2)),opt.y_range(1));
        send_udp_dasher(sprintf([opt.x_label ' %d\n'],curr_data(5)));
        send_udp_dasher(sprintf([opt.y_label ' %d\n'],curr_data(6)));
    else
        curr_data(5) = max(min(curr_data(5),opt.y_range(2)),opt.y_range(1));
        send_udp_dasher(sprintf([opt.y_label ' %d\n'],curr_data(5)));
    end
    if opt.verbosity>1
        curr_data
    end 
    pause(1/opt.fs_out);
end 
    
% close all connections and say goodbye.
get_udp('close');
send_udp_dasher;
disp('No UDP input - Connections closed.');
return