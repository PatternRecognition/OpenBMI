%This directory contains functions that are used in bbci_apply.
%
%BBCI_APPLY_SETDEFAULTS - Set default values in bbci structure for bbci_apply
%BBCI_APPLY_INITDATA - Initialize the data structure of bbci_apply
%BBCI_APPLY_ACQUIREDATA - Fetch data from acquisition hardware
%BBCI_APPLY_EVALSIGNAL - Process cont. acquired signals and store in buffer
%BBCI_APPLY_EVALCONDITION - Evaluate conditions which trigger control signals
%BBCI_APPLY_QUERYMARKER - Check for acquired markers
%BBCI_APPLY_EVALFEATURE - Perform feature extration
%BBCI_APPLY_GETSEGMENT - Retrieve segment of signals from buffer
%BBCI_APPLY_EVALCLASSIFIER - Apply classifier to feature vector
%BBCI_APPLY_EVALCONTROL - Evaluate control function to classifier output
%BBCI_APPLY_SENDCONTROL - Send control signal to application
%BBCI_APPLY_RESETDATA - Reset the data structure of bbci_apply
%BBCI_APPLY_EVALQUITCONDITION - Evalutate whether bbci_apply should stop
