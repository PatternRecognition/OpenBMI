function DS_record= bbci_apply_recordSignals(varargin)
%BBCI_APPLY_RECORDSIGNALS - Save acquired signals to file
%
%Synopsis:
%  DS_REC= bbci_apply_recordSignals('init', BBCI_SOURCE, SOURCE)
%  DS_REC= bbci_apply_recordSignals(SOURCE, MARKER, NMARKERS)
%  DS_REC= bbci_apply_recordSignals('close', DS_REC);

% 02-2012 Benjamin Blankertz


global TODAY_DIR

if ischar(varargin{1}),
  cmd= varargin{1};
  switch(cmd),
   case 'init',
    BS= varargin{2};
    DS= varargin{3};
% This condition is never true, so we comment it out.
%    if ~BS.record_signals,
%      DS_record= struct('recording', 0, 'fcn','');
%      return;
%    end
    DS_record= struct('recording', 1);
    DS_record.opt= propertylist2struct(BS.record_param{:});
    DS_record.opt= set_defaults(DS_record.opt, ...
                                'internal', 0, ...
                                'checkimpedances', 0, ...
                                'folder', TODAY_DIR);
    filebase= BS.record_basename;
    if ~isabsolutepath(filebase),
      filebase= fullfile(DS_record.opt.folder, filebase);
    end
    % Append counter if necessary to avoid overwriting
    num= 1;
    filename= filebase;
    while ~isempty(dir([filename '.*'])),
      num= num + 1;
      filename= sprintf('%s%02d', filebase, num);
    end
    DS_record.filename= filename;
    
    if DS_record.opt.internal,
      DS_record.fcn= 'internal';
    else
      list_external= {'bv'};
      DS_record.fcn= strrep(func2str(BS.acquire_fcn), 'bbci_acquire_', '');
      if ~ismember(DS_record.fcn, list_external),
        warning(['Recording not implemented for ''%s'', ' ...
                 'using internal function'], DS_record.fcn);
        DS_record.fcn= 'internal';
      end
    end
    switch(DS_record.fcn),
     case 'bv',
      if DS_record.opt.checkimpedances,
        bvr_sendcommand('startimprecording', [filename '.eeg']);
      else
        bvr_sendcommand('startrecording', [filename '.eeg']);
      end
     case 'internal',
      opt= copy_fields(DS_record.opt, DS, {'clab','fs'});
      state= acq_recordSignals('init', filename, opt);
      DS_record= copy_fields(DS_record, state);
     otherwise,
      % This should not happen (see 'list_external' above)
      error('Implementation for opening ''%s'' is missing', DS_record.fcn);
    end
    
   case 'close',
    DS_record= varargin{2};
    if ~DS_record.recording,
      return;
    end
    switch(DS_record.fcn),
     case 'bv',
      bvr_sendcommand('stoprecording');
     case 'internal',
      DS_record= acq_recordSignals('close', DS_record);
     otherwise
      error('Implementation for closing ''%s'' is missing', DS_record.fcn);
    end
    
   otherwise,
    error('unknown command');
  end
  return;
end

%if DS_record.recording,
  DS_record= acq_recordSignals(varargin{:});
%end
