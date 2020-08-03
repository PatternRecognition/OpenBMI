% eegTools - general purpose tools for EEG signals
%
% handle data in raw format
%   readGenericEEG          - read data in format of the brain vision recorder
%                             (only multiplexed int16 format)
%                             
%   readGenericEEG_float    - as above but for multiplexed float format
%   readChannelwiseProcessed- read data as with readGenericEEG, but each
%                             a given preprocessing is applied for each
%                             channel separately (for memory critical cases)
%   readGenericHeader       - read information from header file
%   readMarkerTable         - read stimulus and response markers
%   readMarkerComments      - read comment markers
%   readAlternativeMarkers  -
%   readMarkerTableArtifacts
%   writeGenericData        - write data in brain vision format
%   writeChannelPositions   - write channels position for import in
%                             the brain vision recorder
%
% generate marker structure from raw data
%   getDzweiEvents          - see below
%   getPacedEvents          -
%   getPacedSemiimagEvents  -
%   getTriggeredEvents      -
%   getTriggeredKeyEvents   -
%   getTuebingenEvents      - this and the above functions generate markers
%                             taylored to specific experiments
%   getThresholdEvents      - generate markers defined by thresholds of
%                             specific channels
%
% handle data in matlab format ("processed")
%   loadProcessedEEG        - load data (+mrk, mnt) from processed file
%   concatProcessedEEG      - concatenate data and marker structures
%                             from processed files
%   saveProcessedEEG        - save data in matlab format
%
% select subsets of markers
%   mrk_selectEvents        - select a subset of events from a marker
%                             structure, deleting void classes
%   mrk_selectClasses       - select events belonging to given classes
%
% other utilities
%   makeEpochs              - generate epochs (short time segments of data)
%                             from continuous data
%   chanind                 - get indices of given channel labels
%   scalpChannels           - get indices of channels mounted on the scalp
%                             (in contrast to EMG, EOG, ... channels)
%   getBandIndices          - returns indices of fourier bins
%   getIvalIndices          - returns indices of an interval within an epoch
%   addWheelChannels        - adds a 'wheel' and a 'throttle' channel
%                             from the file *_wheel.dat
%                             for our experiments with the steering wheel
%
%
% the following global variables are used. the use of those directories
% can be avoided by specifying absolute file names.
%   EEG_RAW_DIR    - to read files in generic data format 
%   EEG_MAT_DIR    - for files in matlab format
%   EEG_EXPORT_DIR - to export files in generic data format
