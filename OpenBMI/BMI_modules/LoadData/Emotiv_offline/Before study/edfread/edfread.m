function [hdr, record] = edfread(fname, varargin)
% Read European Data Format file into MATLAB
%
% [hdr, record] = edfread(fname)
%         Reads data from ALL RECORDS of file fname ('*.edf'). Header
%         information is returned in structure hdr, and the signals
%         (waveforms) are returned in structure record, with waveforms
%         associated with the records returned as fields titled 'data' of
%         structure record.
%
% [...] = edfread(fname, 'assignToVariables', assignToVariables)
%         Triggers writing of individual output variables, as defined by
%         field 'labels', into the caller workspace.
%
% [...] = edfread(...,'desiredSignals',desiredSignals)
%         Allows user to specify the names (or position numbers) of the
%         subset of signals to be read. |desiredSignals| may be either a
%         string, a cell array of comma-separated strings, or a vector of
%         numbers. (Default behavior is to read all signals.)
%         E.g.:
%         data = edfread(mydata.edf,'desiredSignals','Thoracic');
%         data = edfread(mydata.edf,'desiredSignals',{'Thoracic1','Abdominal'});
%         or
%         data = edfread(mydata.edf,'desiredSignals',[2,4,6:13]);
%
% FORMAT SPEC: Source: http://www.edfplus.info/specs/edf.html SEE ALSO:
% http://www.dpmi.tu-graz.ac.at/~schloegl/matlab/eeg/edf_spec.htm
%
% The first 256 bytes of the header record specify the version number of
% this format, local patient and recording identification, time information
% about the recording, the number of data records and finally the number of
% signals (ns) in each data record. Then for each signal another 256 bytes
% follow in the header record, each specifying the type of signal (e.g.
% EEG, body temperature, etc.), amplitude calibration and the number of
% samples in each data record (from which the sampling frequency can be
% derived since the duration of a data record is also known). In this way,
% the format allows for different gains and sampling frequencies for each
% signal. The header record contains 256 + (ns * 256) bytes.
%
% Following the header record, each of the subsequent data records contains
% 'duration' seconds of 'ns' signals, with each signal being represented by
% the specified (in the header) number of samples. In order to reduce data
% size and adapt to commonly used software for acquisition, processing and
% graphical display of polygraphic signals, each sample value is
% represented as a 2-byte integer in 2's complement format. Figure 1 shows
% the detailed format of each data record.
%
% DATA SOURCE: Signals of various types (including the sample signal used
% below) are available from PHYSIONET: http://www.physionet.org/
%
%
% % EXAMPLE 1:
% % Read all waveforms/data associated with file 'ecgca998.edf':
%
% [header, recorddata] = edfRead('ecgca998.edf');
%
% % EXAMPLE 2:
% % Read records 3 and 5, associated with file 'ecgca998.edf':
%
% header = edfRead('ecgca998.edf','AssignToVariables',true);
% % Header file specifies data labels 'label_1'...'label_n'; these are
% % created as variables in the caller workspace.
%
% Coded 8/27/09 by Brett Shoelson, PhD
% brett.shoelson@mathworks.com
% Copyright 2009 - 2012 MathWorks, Inc.
%
% Modifications:
% 5/6/13 Fixed a problem with a poorly subscripted variable. (Under certain
% conditions, data were being improperly written to the 'records' variable.
% Thanks to Hisham El Moaqet for reporting the problem and for sharing a
% file that helped me track it down.)
% 
% 5/22/13 Enabled import of a user-selected subset of signals. Thanks to
% Farid and Cindy for pointing out the deficiency. Also fixed the import of
% signals that had "bad" characters (spaces, etc) in their names.

% HEADER RECORD
% 8 ascii : version of this data format (0)
% 80 ascii : local patient identification
% 80 ascii : local recording identification
% 8 ascii : startdate of recording (dd.mm.yy)
% 8 ascii : starttime of recording (hh.mm.ss)
% 8 ascii : number of bytes in header record
% 44 ascii : reserved
% 8 ascii : number of data records (-1 if unknown)
% 8 ascii : duration of a data record, in seconds
% 4 ascii : number of signals (ns) in data record
% ns * 16 ascii : ns * label (e.g. EEG FpzCz or Body temp)
% ns * 80 ascii : ns * transducer type (e.g. AgAgCl electrode)
% ns * 8 ascii : ns * physical dimension (e.g. uV or degreeC)
% ns * 8 ascii : ns * physical minimum (e.g. -500 or 34)
% ns * 8 ascii : ns * physical maximum (e.g. 500 or 40)
% ns * 8 ascii : ns * digital minimum (e.g. -2048)
% ns * 8 ascii : ns * digital maximum (e.g. 2047)
% ns * 80 ascii : ns * prefiltering (e.g. HP:0.1Hz LP:75Hz)
% ns * 8 ascii : ns * nr of samples in each data record
% ns * 32 ascii : ns * reserved

% DATA RECORD
% nr of samples[1] * integer : first signal in the data record
% nr of samples[2] * integer : second signal
% ..
% ..
% nr of samples[ns] * integer : last signal

if nargin > 5
    error('EDFREAD: Too many input arguments.');
end

if ~nargin
    error('EDFREAD: Requires at least one input argument (filename to read).');
end

%% hdr

[fid,msg] = fopen(fname,'r');
if fid == -1
    error(msg)
end

assignToVariables = false; %Default
targetSignals = []; %Default
for ii = 1:2:numel(varargin)
    switch lower(varargin{ii})
        case 'assigntovariables'
            assignToVariables = varargin{ii+1};
        case 'targetsignals'
            targetSignals = varargin{ii+1};
        otherwise
            error('EDFREAD: Unrecognized parameter-value pair specified. Valid values are ''assignToVariables'' and ''targetSignals''.')
    end
end


% HEADER
hdr.ver        = str2double(char(fread(fid,8)'));
hdr.patientID  = fread(fid,80,'*char')';
hdr.recordID   = fread(fid,80,'*char')';
hdr.startdate  = fread(fid,8,'*char')';% (dd.mm.yy)
% hdr.startdate  = datestr(datenum(fread(fid,8,'*char')','dd.mm.yy'), 29); %'yyyy-mm-dd' (ISO 8601)
hdr.starttime  = fread(fid,8,'*char')';% (hh.mm.ss)
% hdr.starttime  = datestr(datenum(fread(fid,8,'*char')','hh.mm.ss'), 13); %'HH:MM:SS' (ISO 8601)
hdr.bytes      = str2double(fread(fid,8,'*char')');
reserved       = fread(fid,44);
hdr.records    = str2double(fread(fid,8,'*char')');
hdr.duration   = str2double(fread(fid,8,'*char')');
% Number of signals
hdr.ns    = str2double(fread(fid,4,'*char')');
for ii = 1:hdr.ns
    hdr.label{ii} = regexprep(fread(fid,16,'*char')','\W','');
end

if isempty(targetSignals)
    targetSignals = 1:numel(hdr.label);
elseif iscell(targetSignals)||ischar(targetSignals)
    targetSignals = find(ismember(hdr.label,regexprep(targetSignals,'\W','')));
end
if isempty(targetSignals)
    error('EDFREAD: The signal(s) you requested were not detected.')
end

for ii = 1:hdr.ns
    hdr.transducer{ii} = fread(fid,80,'*char')';
end
% Physical dimension
for ii = 1:hdr.ns
    hdr.units{ii} = fread(fid,8,'*char')';
end
% Physical minimum
for ii = 1:hdr.ns
    hdr.physicalMin(ii) = str2double(fread(fid,8,'*char')');
end
% Physical maximum
for ii = 1:hdr.ns
    hdr.physicalMax(ii) = str2double(fread(fid,8,'*char')');
end
% Digital minimum
for ii = 1:hdr.ns
    hdr.digitalMin(ii) = str2double(fread(fid,8,'*char')');
end
% Digital maximum
for ii = 1:hdr.ns
    hdr.digitalMax(ii) = str2double(fread(fid,8,'*char')');
end
for ii = 1:hdr.ns
    hdr.prefilter{ii} = fread(fid,80,'*char')';
end
for ii = 1:hdr.ns
    hdr.samples(ii) = str2double(fread(fid,8,'*char')');
end
for ii = 1:hdr.ns
    reserved    = fread(fid,32,'*char')';
end
hdr.label = hdr.label(targetSignals);
hdr.label = regexprep(hdr.label,'\W','');
hdr.units = regexprep(hdr.units,'\W','');
disp('Step 1 of 2: Reading requested records. (This may take a few minutes.)...');
if nargout > 1 || assignToVariables
    % Scale data (linear scaling)
    scalefac = (hdr.physicalMax - hdr.physicalMin)./(hdr.digitalMax - hdr.digitalMin);
    dc = hdr.physicalMax - scalefac .* hdr.digitalMax;
    
    % RECORD DATA REQUESTED
    tmpdata = struct;
    for recnum = 1:hdr.records
        for ii = 1:hdr.ns
            % Read or skip the appropriate number of data points
            if ismember(ii,targetSignals)
                % Use a cell array for DATA because number of samples may vary
                % from sample to sample
                tmpdata(recnum).data{ii} = fread(fid,hdr.samples(ii),'int16') * scalefac(ii) + dc(ii);
            else
                fseek(fid,hdr.samples(ii)*2,0);
            end
        end
    end
    hdr.units = hdr.units(targetSignals);
    hdr.physicalMin = hdr.physicalMin(targetSignals);
    hdr.physicalMax = hdr.physicalMax(targetSignals);
    hdr.digitalMin = hdr.digitalMin(targetSignals);
    hdr.digitalMax = hdr.digitalMax(targetSignals);
    hdr.prefilter = hdr.prefilter(targetSignals);
    hdr.transducer = hdr.transducer(targetSignals);
    
    record = zeros(numel(hdr.label), hdr.samples(1)*hdr.records);
    % NOTE: 5/6/13 Modified for loop below to change instances of hdr.samples to
    % hdr.samples(ii). I think this underscored a problem with the reader.
    
    disp('Step 2 of 2: Parsing data...');
    recnum = 1;
    for ii = 1:hdr.ns
        if ismember(ii,targetSignals)
            ctr = 1;
            for jj = 1:hdr.records
                try
                    record(recnum, ctr : ctr + hdr.samples(ii) - 1) = tmpdata(jj).data{ii};
                end
                ctr = ctr + hdr.samples(ii);
            end
            recnum = recnum + 1;
        end
    end
    hdr.ns = numel(hdr.label);
    hdr.samples = hdr.samples(targetSignals);

    if assignToVariables
        for ii = 1:numel(hdr.label)
            try
                eval(['assignin(''caller'',''',hdr.label{ii},''',record(ii,:))'])
            end
        end
        record = [];
    end
end

% hdr.label';
% hdr.NS=length(hdr.label);
% hdr=leadidcodexyz(hdr) %% read channel location



fclose(fid);