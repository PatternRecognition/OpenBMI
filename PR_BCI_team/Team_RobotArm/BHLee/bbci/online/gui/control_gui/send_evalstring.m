function send_evalstring(machine,port,string);
% SEND_EVALSTRING ONLY FOR INTERN USE OF MATLAB_CONTROL_GUI
%
% send the evaluable string to the machine at port 
%
% usage:
%   send_evalstring(machine,port,string);
%
% input:
%   machine   name of a machine
%   port      port number
%   string    evaluable string (is not checked)
%
% Guido Dornhege
% $Id: send_evalstring.m,v 1.1 2006/04/27 14:21:11 neuro_cvs Exp $

  
send_data_udp(machine,port,double(string));

fprintf('send to %s (%d):\n',machine,port);
fprintf('%s\n\n',string);








