%This folder contains the so-called CONTROL functions for bbci_apply.
%These functions transform the classifier output into the control signal
%(PACKET), that will be sent to the application via UDP. The PACKET is 
%formatted as a variable/value list in a CELL.
%
%These functions have the following format:
%
% --- --- --- ---
%BBCI_CONTROL_XYZ - Generate control signal for application XYZ
%
%Synopsis:
%  [PACKET, STATE]= bbci_control_XYZ(CFY_OUT, STATE, EVENT_INFO, <PARAMS>)
%
%Arguments:
%  CFY_OUT - Output of the classifier
%  STATE - Internal state variable, which is empty in the first call of a
%      run of bbci_apply.
%  EVENT_INFO - Structure that specifies the event (fields 'time' and 'desc')
%      that triggered the evaluation of this control. Furthermore, EVENT_INFO
%      contains the whole marker queue in the field 'all_markers'.
%  PARAMS - Additional parameters that can be specified in bbci.control.param
%      are passed as further arguments.
%
%Output:
% PACKET: Variable/value list in a CELL defining the control signal that
%     is to be sent via UDP to the application.
% STATE: Updated internal state variable
% --- --- --- ---
%
%
%List of CONTROL functions (prefix bbci_control_ is left out):
% ERP_Speller: ERP-based Hex-o-Spell, one output for each complete sequence
% ERP_Speller_binary: ERP-based Hex-o-Spell, one outputs for each stimulus
