function mrk= mrkodef_LCT2_foot_visu(mrko, opt)

% mrkodef_LCT2_foot_visu - prepare markers for visual LCT2 experiment
%
%Synopsis:
% mrk= mrkdef_LCT2_foot_visu(mrko, <OPT>)
%
%Arguments:
% mrk = mrk structure as given in output from eegfile_readBVmarkers
% file - name of the eeg data file without file type ending, relative to         EEG_RAW_DIR
% OPT  - struct or property/value list of optional properties
% 
%Output:
% MRK - preprocessed narker structure with the following fields
%       .pos position of a marker (in number of samples)
%       .toe original coding of the mrks
%       .fs sample frequency
%       .y class assignment to classes named in:
%       .className (sequence is kept in .y columnwise) : {'tar' 'nontar'}
%       .ishit the stimulus was correctly identified as target or notarget
%       .duration time in msec from the stimulus presentation until the motor response
%       .add_res_pos positions of additional responses, when a response was already given within one run
%       .indexedByEpochs: contains mrk fields that have to be accounted for when making epochs.
%%
% Marker codings:
% 252: begin
% 253: end1
% 254: end2
%  64: laser ON
% 101: laser OFF
%  -2: RT, i.e. break
% -64: gas, i.e. accelerate
%  32: marker, stay or go to the left side
%  16: marker, stay or go to the center
%   8: marker, stay or go to the right side
%  48: curve, start
%  24: curve, end
% -66: RT + gas
%  80: Laser ON + maker center
%  96: Laser ON + marker left
%  72: Laser ON + marker right
%See:
%   eegfile_readBVmarkers mrkodef_imag_fb01

%Author(s) Claudia Sannelli, September-2008
% 
% stim = [64, 72, 80, 96];
% for s = 1:length(stim)
%     if ~isempty(find(mrko.toe == stim(s)))
%         stim_vec = [stim_vec stim(s)];
%         stim_str = 
    
stimDef = {'S 64', 'S 72', 'S 80', 'S 96', 'S112', 'S 88'; 'laserON', 'laserON+goRight', 'laserON+goCenter', 'laserON+goLeft', 'laserON+curveStart', 'laserON+curveEnd'};
respDef = {{'R  2', 'R 66'}; 'break'};
miscDef = {'R 64', 'S 8', 'S 16', 'S 32', 'S 24', 'S 48', 'S101', 'S109', 'S117', 'S133', 'S149', 'S125', 'S252', 'S253', 'S254'; ...
  'gas', 'goRight', 'goCenter', 'goLeft', 'curveStart', 'curveEnd', 'laserOFF', 'laserOFF+goRight', 'laserOFF+goCenter', ...
  'laserOFF+goLeft', 'laserOFF+curveStart', 'laserOFF+curveEnd', 'init', 'timeout', 'timout'};
curveInDef = {'S 24', 'S112'; 'curveStart', 'laserON+curveStart'};
curveOutDef = {'S 48', 'S 88'; 'curveEnd', 'laserON+curveEnd'};

mrk_stim = mrk_defineClasses(mrko, stimDef);
mrk_resp = mrk_defineClasses(mrko, respDef);
mrk_misc = mrk_defineClasses(mrko, miscDef);
mrk_curveIn = mrk_defineClasses(mrko, curveInDef);
mrk_curveOut = mrk_defineClasses(mrko, curveOutDef);

fprintf('%d trials rejected: warmup\n', opt.adapt_trials);
idxInit = find(mrk_misc.toe == 252);
idxTimeout = find(mrk_misc.toe == 253);

% in VPef, strange, end-markers are missed, for this reason check at first
% whether start and end markers exist
if ~isempty(idxInit)
    posInit = mrk_misc.pos(idxInit);
else
    posInit = mrko.pos(1);
end

if ~isempty(idxTimeout)
    posTimeout = mrk_misc.pos(idxTimeout);
else
    posTimeout = mrko.pos(end);
end

idxIn = intersect(find(mrk_stim.pos>posInit),find(mrk_stim.pos<posTimeout));
mrk_stim = mrk_chooseEvents(mrk_stim, idxIn);

idxIn = intersect(find(mrk_resp.pos>posInit),find(mrk_resp.pos<posTimeout));
mrk_resp = mrk_chooseEvents(mrk_resp, idxIn);

mrk_stim = mrk_chooseEvents(mrk_stim, opt.adapt_trials+1:length(mrk_stim.pos));
mrk = mrk_matchStimWithResp(mrk_stim, mrk_resp, 'removevoidclasses', 0, 'missingresponse_policy', 'accept', 'multiresponse_policy', 'first');

mrk.icurve = zeros(1,length(mrk.pos));

if ~isempty(mrk_curveIn.pos) && isempty(mrk_curveOut.pos)
    disp('Curve starts and not end, check better')
    keyboard
elseif ~isempty(mrk_curveIn.pos) && ~isempty(mrk_curveOut.pos)
    [mrk_curve, iIn, iOut] = mrk_matchStimWithResp(mrk_curveIn, mrk_curveOut, 'removevoidclasses', 1, 'missingresponse_policy', 'reject', 'multiresponse_policy', 'first');
    for i = 1:length(iIn)
        idx1 = min(find(mrk.pos >= mrk_curveIn.pos(iIn(i))));
        idx2 = min(find(mrk.pos <= mrk_curveOut.pos(iOut(i))));
        if ~isempty(idx1) && ~isempty(idx2)
            mrk.icurve(idx1:idx2) = 1;
        end
    end
end

mrk.reac = mrk.latency;
mrk = rmfield(mrk, 'latency');

mrk.ishit = ismember(mrk.resp_toe, -2);
mrk.imiss = ismember(mrk.resp_toe, 'NaN');
mrk = rmfield(mrk, 'resp_toe');
mrk.indexedByEpochs = {'reac', 'ishit', 'imiss', 'multiresponse', 'icurve'};
mrk.misc = mrk_misc;
