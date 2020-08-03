function [source, marker]= bbci_apply_acquireData(source, bbci_source, marker)
%BBCI_APPLY_ACQUIREDATA - Fetch data from acquisition hardware
%
%Synopsis:
%  SOURCE= bbci_apply_acquireData(SOURCE, BBCI_SOURCE);
%  [SOURCE, MARKER]= bbci_apply_acquireData(SOURCE, BBCI_SOURCE, MARKER);
%
%Arguments:
%  SOURCE - Structure of recently acquired signals and state information
%     about the acquisition setting; subfield of 'data' structure of
%     bbci_apply.
%  BBCI_SOURCE - Structure specifying the settings of acquiring signals
%     from a data source; subfield of 'bbci' structure of bbci_apply.
%  MARKER - Structure holding the recently received markers.
%     The length of the queue of stored markers is defined in
%     bbci.marker.queue_length and set in SOURCE by the function
%     bbci_apply_initData.
%For a description of the fields of these structures, type
%'help bbci_apply_structures'.
%
%Output:
%  SOURCE - Updated strcture of incoming signals
%  MARKER - Updated marker structure

% 02-2011 Benjamin Blankertz


source.x= [];
%%%
mrkDesc_min1 = []; %%%%%%%%%% avoid doubleclicks
mrkDesc_min2 = []; %%%%%%%%%% avoid doubleclicks
if exist('bbci_source.skip_doubleclicks','var') == 0
	bbci_source.skip_doubleclicks = false;
end
%%%
run= 1;
while run,
  look_for_data= true;
  while look_for_data,
    [new_data, mrkTime, mrkDesc, source.state] = ...
        bbci_source.acquire_fcn(source.state);
    look_for_data= isempty(new_data) && bbci_source.min_blocklength_sa>0 && ...
        source.state.running;
  end
  
  nMarkers= length(mrkTime);
  if nMarkers>0,
    % transfer relative marker positions (within new block)
    % to absolute marker positions relative to start
    mrkTime= mrkTime + source.time;
    if ~isempty(bbci_source.marker_mapping_fcn),
      [mrkDesc, idx]= bbci_source.marker_mapping_fcn(mrkDesc);
      mrkTime= mrkTime(idx);
      nMarkers= length(idx);
    end
%%%
    if bbci_source.skip_doubleclicks 
        if ~(mrkDesc_min2 == mrkDesc) 
            marker.time= cat(2, marker.time(nMarkers+1:end), mrkTime);
            marker.desc= cat(2, marker.desc(nMarkers+1:end), mrkDesc);
        end 
		mrkDesc_min2 = mrkDesc_min1; 
		mrkDesc_min1 = mrkDesc; 
		%marker.desc
    else 
        marker.time= cat(2, marker.time(nMarkers+1:end), mrkTime);
        marker.desc= cat(2, marker.desc(nMarkers+1:end), mrkDesc);
    end
%%%
  end 
  source.x= cat(1, source.x, new_data);
  source.sample_no= source.sample_no + size(new_data,1);
  source.time= source.sample_no*1000/source.fs;
  run= (size(source.x,1) < bbci_source.min_blocklength_sa);
end

if size(source.x,1) > bbci_source.min_blocklength_sa,
  bbci_apply_logger(source.log.fid, ...
                    '# Source: block length %d samples.\n', size(source.x,1));
end
