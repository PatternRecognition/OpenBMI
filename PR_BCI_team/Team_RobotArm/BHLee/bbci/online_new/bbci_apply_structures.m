%In BBCI_APPLY there are two central structures. 
%
%- The 'bbci' structure specifies WHAT should be done and HOW:
%data acquisition, processing, feature extraction, classification,
%determining the control signal, and calling the application.
%It is the input to bbci_apply.
%
%- The 'data' structure is used to store the acquired signals, and various
%steps of processed data, as well as some state information.
%It is the working variable of bbci_apply.
%
%
%* Structure BBCI (defaults are set in bbci_apply_setDefaults):
%
%bbci.source - Defines the sources for acquiring signals
%  * struct array with fields:
%  .acquire_fcn   [STRING, default 'acquire_bv']
%  .acquire_param [CELL ARRAY, default {}]: parameters to acquire_fcn
%  .min_blocklength [DOUBLE, default 40] minimum blocklength [msec] that should
%                 be acquired before dat is passed to further processing
%                 (in bbci_apply_setDefaults a variant .min_blocklength_sa
%                 is added for convenience.)
%  .clab          [CELL ARRAY of STRING, default {'*'}]
%  .log           see bbci.log. This field specifies, whether source-specific
%                 information should be logged (which is reporting when
%                 the length of an acquired block is larger than
%                 .min_blocklength)
%bbci.marker - Defines how the acquired markers are stored
%  * struct with fields:
%  .queue_length  Specifies how many markers are stored in the marker queue
%                 (see data.marker). The markers in the queue are available
%                 for queries and evaluating conditions, see
%                 bbci_apply_queryMarker.
%bbci.signal - Defines how the continuous signals are preprocessed and
%                 stored into the ring buffer
%  * struct array with fields:
%  .source        [DOUBLE, default 1]
%  .proc          [CELL ARRAY, one cell per proc function,
%                  each CELL is either a STRING, or a
%                  CELL ARRAY {FUNC, PARAM}, where FUNC is a STRING and
%                  PARAM is a CELL ARRAY of parameters to the function;
%                  default {}]
%  .buffer_size   [DOUBLE, default 5000] in msec
%  .clab          [CELL ARRAY of STRING, default {'*'}]
%bbci.feature - Defines extraction of features from continuous signals
%  * struct array with fields:
%  .signal        [vector of DOUBLE, default 1]
%  .ival          vector [start_msec end_msec]
%  .proc          [CELL ARRAY, one cell per proc function,
%                  each CELL is either a STRING, or a
%                  CELL ARRAY {FUNC, PARAM}, where FUNC is a STRING and
%                  PARAM is a CELL ARRAY of parameters to the function;
%                  default {}]
%bbci.classifier - Specifies classification (model and parameters)
%  * struct array with fields:
%  .feature       [vector of DOUBLE, default 1]
%  .apply_fcn
%  .C
%bbci.control - Defines how to translate the classifier output (and given the
%               event marker) into the control signal
%  * struct array with fields:
%  .classifier    [vector of DOUBLE, default 1]
%  .fcn           [STRING, default '']
%  .param         (if ~isempty(bbci.control.fcn))
%  .condition     defines the events which evokes the calculation of a
%                 control signal:
%                 [] means evaluate control signal for each data packet
%                 that was acquired
%     .marker        CELL of STRINGs (??or rather [vector of DOUBLE]??)
%                    specifying the markers that evoke the calculation of a
%                    control signal (if
%     .interval      [DOUBLE in msec] (does this option make sense?)
%     .overrun       [DOUBLE in msec] after .marker this amount of signals must
%                    have been required (such that epochs of all required
%                    .feature can be obtained)
%bbci.feedback - Defines where and how the control signal is sent
%  * struct array with fields:
%  .control       [vector of DOUBLE, default 1] numbers of the control signals
%                 that are send to the feedback application
%  .receiver      'matlab', 'pyff', 'screen' (default), or 'tobi-c'
%bbci.adaptation - Specifies whether, what and how adaptation should be done
%  * struct with fields
%  .active        [BOOL, default 1] whether adaptation is switched on.
%  .fcn           CHAR name of adaptation function ('bbci_adaptation_'
%                 is prepended).
%  .param         CELL parameters that are passed to the adaptation.fcn
%  .log           see bbci.log. This field specifies, whether information about
%                 adaptation should be logged
%bbci.quit_condition - Defines the condition when bbcu_apply should quit
%  * struct with fields
%  .running_time  [DOUBLE in sec, default inf]
%  .marker        [CHAR or CELL ARRAY of CHAR, default '']
%bbci.log - Defines whether and how information should be logged
%  .output        0 (or 'none') for no logging, or 'screen', or 'file',
%                 or 'screen&file';
%                 'screen' is default if bbci.feeback.receiver is empty,
%                 otherwise 0.
%  .folder        CHAR folder for storing logfiles (if requested)
%  .file          CHAR filename of logfile.
%  .time_fmt      CHAR print format of the time, default '%08.3fms'
%  .clock         BOOL specifies whether the clock should also be logged,
%                 default 0.
%  .classifier    BOOL specifies whether the classifer should also be logged,
%                 default 0.
%  .force_overwriting BOOL: if true, the log file is always saved under the
%                 specified filename, even if the file already exists; 
%                 otherwise, the value of an incremental counter is added in 
%                 order to avoid overwriting.
%
%Optionally further features:
%  remote_control (let parameters be changed over UDP, e.g. by a GUI)?
%
%
%  --- --- --- --- --- --- --- ---
%
%
%* Structure DATA  (initialized in bbci_apply_initData.m):
%
%data.source - struct array with fields:
%  .state        state structure of acquire function
%  .x            recent block of acquired data
%  .fs           sampling rate
%  .clab         CELL of channel labs in source.x
%                (these are selected by bbci.source.clab)
%  .sample_no    number of the last sample in the recent data (source.x)
%                relative to the start of bbci_apply
%  .time         time of acquisition, i.e. 'sample_no' converted to msec
%data.marker - struct with fields:
%  .time         [DOUBLE: 1xBBCI.MARKER.QUEUELENGTH] in msec(!) since start
%  .desc         [CELL: 1xBBCI.MARKER.QUEUELENGTH of STRINGs] marker
%                descriptors OR
%                [DOUBLE: 1xBBCI.MARKER.QUEUELENGTH] numeric format of
%                marker descriptors 
%  .current_time [DOUBLE] time of last acquired sample since start in msec
%data.signal - struct array with fields:
%  .size         Size of the buffer (in time dimension) in unit samples.
%  .x            [DOUBLE: TIMExCHANNELS] storing the recent continuous
%                signals as a ring buffer. This buffer needs to be large
%                enough (set by bbci.signal.buffer_size) to hold
%                segments from which features are calculated, see
%                bbci.feature.ival.
%  .ptr          Points to the last stored sample (in time dimension).
%  .clab         Labels of the channels in the signal.
%  .fs           sampling rate
%  .use_state    [BOOLEAN: nFcns] For each function in bbci.signal.fcn
%                this flag indicates whether it uses state variables.
%  .state        CELL used to store states of the bbci.signal.fcn functions
%  .time         [DOUBLE] time of last acquired sample since start in msec
%data.feature - struct array with fields:
%  .x            [DOUBLE: FEAT_DIMx1] Feature vector
%  .time         reference time to which the feature is related
%data.classifier - struct array with fields:
%  .x
%data.control - Control signal to be sent to the application via UDP
%               (or passed as argument in a direct call of a Matlab feedback)
%  * struct array with fields:
%  .lastcheck    Time of the last condition check wrt. to this control function
%  .state        Field that can be used by control functions to store
%                'persistent' variables.
%  .packet       Cell specifying a variable/value list;
%data.log - Information needed for logging
%  .fid          file ID of log file (or 1 is bbci.log.output=='screen'),
%                if bbci.log.output=='screen&file', this is a vector
%                [1 file_id].
%  .filename     name of the log file (if bbci.log.output is 'file' or
%                'screen&file')

