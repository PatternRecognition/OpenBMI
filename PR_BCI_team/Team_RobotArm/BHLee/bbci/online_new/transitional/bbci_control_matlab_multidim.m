function packet= bbci_control_matlab_multidim(cfy_out, varargin)

%BBCI_CONTROL_MATLAB_MULTIDIM - Generate control signal 
%
%Synopsis:
%  PACKET= bbci_control_matlab_multidim(CFY_OUT, <EVENT_INFO>)
%
%Arguments:
%  CFY_OUT - Output of the classifier
%  EVENT_INFO - Structure that specifies the event (fields 'time' and 'desc')
%      that triggered the evaluation of this control. Furthermore, EVENT_INFO
%      contains the whole marker queue in the field 'all_markers'.
%
%Output:
% PACKET: Variable/value list in a CELL defining the control signal that
%     is to be sent via UDP to the application.

% 01-2012 Benjamin Blankertz


sz= numel(cfy_out);
packet= cat(1, repmat({'cl_output'},[1 sz]), num2cell(cfy_out(:)'));
packet= reshape(packet, [1, 2*sz]);
