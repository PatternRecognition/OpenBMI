

function [eye mrk] = eye_synchWithEEG(mrk, eye, cnt, varargin)
%
% Synchronizes EEG and ET data. EEG data is left unchanged; ET data is
% interpolated at the timepoints of the EEG data. The resulting ET data
% will thus have the same sampling frequency as the EEG data.
%
% USAGE:
%   eye = function synch_eeg_eye(mrk, eye, varargin)
%
% IN:       mrk         -       EEG marker struct
%           eye         -       ET struct
%           cnt         -       EEG continuous data struct
%
% OUT:      eye         -       updated ET struct
%
%
% Simon Scholler, 2011
%

opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...    
                   'interpolation', 'linear', ...        % cf. interp1.m
                   'synch_mrk', 'SyncMarker', ...        % description string for the synchronization marker in the eeg-mrk-struct
                   'vars', {'L Raw X [px]'    'L Raw Y [px]'    'L Dia X [px]'    'L Dia Y [px]'    'L POR X [px]'    'L POR Y [px]'}, ...  % data variables to be kept
                   'save_orig', 0, ...   % save the original data in field .orig
                   'visualize', 0);   % plot inter-marker offset histogram?


if ~isstruct(mrk)   % input are filenames
  eeg_fnames = mrk;
  eye_fnames = eye;
  [cnt mrk] = eegfile_loadBV(eeg_fnames);  
  eye = eye_readIDF(eye_fnames);
end

%% Extract synch-markers from marker struct
ev_synch = strmatch(opt.synch_mrk, mrk.desc);
mrk = mrk_selectEvents(mrk, ev_synch);
mrk.t = (mrk.pos-mrk.pos(1)) * 1000 / mrk.fs;  % convert pos to msec


%% Assign markers if marker structs have different length
len_eeg = length(mrk.t);
len_eye = length(eye.mrk.t);

if len_eeg~=len_eye 
%% Remove markers at the beginning/end (if necessary) and align remaining markers of ET and EEG 
   opt.visualize = 1;
   warning('Different number of markers in the EEG and ET mrk struct. Trying to assign markers to each other... ')
   
   % Remove markers at the beginning/end (if necessary)
   if mrk.t(1)<eye.mrk.t(1)
       [tmp idx] = min(abs(mrk.t-eye.mrk.t(1)));
       mrk.t = mrk.t(idx:end);
   elseif mrk.t(1)>eye.mrk.t(1)
       [tmp idx] = min(abs(eye.mrk.t-mrk.t(1)));
       eye.mrk.t = eye.mrk.t(idx:end);       
   end
   if mrk.t(end)<eye.mrk.t(end)
       [tmp idx] = min(abs(eye.mrk.t-mrk.t(end)));
       eye.mrk.t = eye.mrk.t(1:idx);
   elseif mrk.t(1)>eye.mrk.t(1)
       [tmp idx] = min(abs(mrk.t-eye.mrk.t(end)));
       mrk.t = mrk.t(1:idx);       
   end
   
   % Align remaining markers of ET and EEG
   len_eeg = length(mrk.t);
   len_eye = length(eye.mrk.t);
   if len_eeg~=len_eye      
       if len_eeg>len_eye
           shorter = eye.mrk.t;
           longer = mrk.t;
       else
           shorter = mrk.t;
           longer = eye.mrk.t;
       end
       sel = [];
       for n = 1:length(shorter)
           [tmp idx] = min(abs(longer-shorter(n)));
           sel(n) = idx;
       end
       if length(sel) ~= length(unique(sel))
           error('Assigning markers failed.')
       end
       if len_eeg>len_eye
           mrk.t = mrk.t(sel);
       else
           eye.mrk.t = eye.mrk.t(sel);
       end
   end
   disp('Assignment succesful (opt.visualize has been set ''on'' to check the result of the marker assignment).')
end


%% Stretch or compress timeaxis of ET recording according to start/end marker alignment (if necessary)
T_eeg = diff(mrk.t([1 end]));
T_et = diff(eye.mrk.t([1 end]));
%warp = T_eeg/T_et;
out = polyfit(eye.mrk.t,mrk.t,1);
warp = out(1);
shift = out(2);
eye.mrk.t = eye.mrk.t * warp;  % warp eyetracker-timeline


if opt.visualize
   figure, hist(mrk.t-eye.mrk.t), title(['EEG-ET sync marker time difference histogram (Mean offset: ' num2str(shift,'%.2f') 'ms)'])
   xlabel('Time difference [msec]')
end


%% Interpolate ET data at EEG sample timepoints
T_et_data = eye.t * warp;
T_eeg_data = 1:1000/cnt.fs:size(cnt.x,1)*1000/cnt.fs;  % EEG data has a well-defined sampling frequency
eye.orig = eye;
eye.dat = zeros(length(T_eeg_data),length(opt.vars));
eye.variables = opt.vars;

%figure
for v=1:length(opt.vars)
   idx = strmatch(opt.vars{v}, eye.orig.variables);
   d = cell2mat(eye.orig.dat(:,idx));
   %plot(d), hold on 
   eye.dat(:,v) = interp1(T_et_data , d , T_eeg_data , opt.interpolation);
end

eye.fs = mrk.fs;
eye.t = T_eeg_data;

if ~opt.save_orig
   eye = rmfield(eye,'orig');
end

%figure, subplot(2,1,1), plot(eye.dat), subplot(2,1,2), plot(cnt.x)

