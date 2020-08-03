%This folder contains the so-called ACQUIRE functions for bbci_apply.
%These functions acquire small bolcks of signals (and maybe event markers
%from a specific measurement device, and provide it for online processing.
%If no new data is available, they return an empty structure.
%
%These functions have the following format:
%
% --- --- --- ---
%BBCI_ACQUIRE_XYZ - Online data acquisition from device XYZ
%
%Synopsis:
%  STATE= bbci_acquire_XYZ('init', <PARAM>)
%  [CNTX, MRKTIME, MRKDESC, STATE]= bbci_acquire_XYZ(STATE)
%  bbci_acquire_XYZ('close')
%  bbci_acquire_XYZ('close', STATE)
% 
%Arguments:
%  PARAM - Optional arguments, specific to XYZ.
%  
%Output:
%  STATE - Structure characterizing the incoming signals; fields:
%     'fs', 'clab', and intern stuff
%  CNTX - 'acquired' signals [Time x Channels]
%  The following variables hold the markers that have been 'acquired' within
%  the current block (if any).
%  MRKTIME - DOUBLE: [1 nMarkers] position [msec] within data block.
%      A marker occurrence within the first sample would give
%      MARTIME= 1/STATE.fs.
%  MRKDESC - CELL {1 nMarkers} descriptors like 'S 52'
% --- --- --- ---
%
%List of ACQUIRE functions (prefix bbci_acquire_ is left out)
% bv:      Acquire data from BV Recorder (option 'Remote Data Access' must
%          be enabled in the BV Recorder settings!)
% nirx:    Acquire data from a NIRx system
% offline: Simulate online acquisition by returning small chunks of signals
%          from an initially given data file.
% randomSignals: Generate random signals
