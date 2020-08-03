function [data, bbci]= bbci_apply_initData(bbci)
%BBCI_APPLY_INITDATA - Initialize the data structure of bbci_apply
%
%This function initializes the central 'data' structure of bbci,
%which holds the recent data, markers, extracted features, classifier
%output and some state information.
%In particular, the acquire function is called to initialize the
%acquisition of the signals, and to get information about the settings
%e.g., the channel labels.
%
%Synopsis:
%  [DATA, BBCI]= bbci_apply_initData(BBCI)

% BBCI has to be output argument here, because initialization of the
% adaptation may alter bbci.

% 02-2011 Benjamin Blankertz


%% Initialize 'marker'
data.marker= struct;
data.marker.time= NaN*ones(1, bbci.marker.queue_length);
data.marker.desc= [];
%Now, we do this in bbci_apply_acquireData.m
%if strcmp(bbci.marker.format, 'numeric'),
%  data.marker.desc= NaN*ones(1, bbci.marker.queue_length);
%else
%  data.marker.desc= cell(1, bbci.marker.queue_length);
%end

%% Initialize 'source'
BS= bbci.source;
%data.source= repmat(struct, [length(BS) 1]);
header_str= '';
for k= 1:length(BS),
  % init acquisition source and get information
  DS= struct('x',[]);
  %DS.state= BS(k).acquire_fcn('init', BS(k).acquire_param);
  DS.state= BS(k).acquire_fcn('init', BS(k).acquire_param{:});
  DS.state.running= 1;
  DS.state.chan_sel= chanind(DS.state.clab, BS(k).clab);
  DS.clab= DS.state.clab(DS.state.chan_sel);
  DS.fs= DS.state.fs;
  DS.sample_no= 0;
  DS.time= 0;
  if BS(k).record_signals,
    DS.record= bbci_apply_recordSignals('init', BS(k), DS);
    % Construct a string that will be written into the log file.
    % This string points to the file(s) holding the recorded signals.
    header_str= strcat(header_str, '#Source file');
    if length(BS)>1,
      header_str= strcat(header_str, sprintf(' (%d)', k));
    end
    header_str= [header_str ': ' DS.record.filename];
    if k<length(BS),
      header_str= sprintf('%s\n', header_str);
    end
  else
    DS.record= struct('recording', 0, 'fcn','');
  end
  data.source(k)= DS;
end
if ~isempty(header_str),
  bbci= bbci_log_setHeaderInfo(bbci, header_str);
end
for k=1:length(bbci.source)
  [bbci.source(k).min_blocklength_sa]= ...
      deal(bbci.source(k).min_blocklength.*data.source(k).fs/1000);
end

%% Initialize 'signal'
BS= bbci.signal;
%data.signal= repmat(struct, [length(BS) 1]);
for k= 1:length(BS),
  % init acquisition source and get information
  DB= struct;
  DS= data.source(BS(k).source);
  DB.size= ceil(BS(k).buffer_size/1000*DS.state.fs);
  DB.ptr= 0;
  % Get channel indices here and store them to speed up bbci_apply
  DB.chidx= chanind(DS, BS(k).clab);
  % Prepare cnt structure for efficiency
  DB.cnt= struct('fs', DS.fs, 'clab',{DS.clab(DB.chidx)});
  % We cannot allocate memory for the buffer, since we do not know,
  % how many channels remain after applying 'bbci.signal'.
  DB.x= [];
  DB.fs= 0;
  DB.clab= {};
  % Check for each signal function, whether it uses a state variable
  DB.use_state= zeros(1, length(BS(k).fcn));
  for j= 1:length(BS(k).fcn),
    % The following check might lead to 'false positives': There are some
    % proc_* functions that have more than 1 output, but do not use a
    % state variable.
    if nargout(BS(k).fcn{j})>1,
      DB.use_state(j)= 1;
    end
  end
  DB.state= repmat({[]}, [1 length(BS(k).fcn)]);
  DB.time= 0;
  data.signal(k)= DB;
end

%% Initialize 'feature', 'classifier', and 'control'
data.feature= struct('x', repmat({[]}, [length(bbci.feature) 1]));
[data.feature.time]= deal(-inf);

data.classifier= struct('x', repmat({[]}, [length(bbci.classifier) 1]));

void_control= struct('packet',[], 'state',[], 'lastcheck',0);
data.control= repmat(void_control, [length(bbci.control) 1]);

%% Initialize 'feedback'
for k= 1:length(bbci.feedback),
  bbci_apply_sendControl('init', bbci.feedback(k));
end

%% Initialite logging
data.log= bbci_log_open(bbci.log);
bbci_prettyPrint(data.log.fid, bbci);
src_log= bbci_log_open(bbci.source(1).log, data.log.fid);
[data.source.log]= deal(src_log);
ada_log= bbci_log_open(bbci.adaptation(1).log, data.log.fid);
ada_log.time_fmt= bbci.log.time_fmt;
data.adaptation= repmat({struct('log',ada_log)}, [1 length(bbci.adaptation)]);

%% Initialize adaptation
[bbci, data]= bbci_apply_adaptation(bbci, data, 'init');
