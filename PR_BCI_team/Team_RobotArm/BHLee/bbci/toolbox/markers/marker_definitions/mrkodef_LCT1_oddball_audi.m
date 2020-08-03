function mrk= mrkodef_LCT1_oddball_audi(mrko, opt)

% mrkodef_LCT1_oddball_audi - prepare markers for visual LCT2 experiment
%
%Synopsis:
% mrk= mrkdef_LCT1_oddball_audi(mrko, <OPT>)
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
% Marker codings:[1 2 -8 -16 -24 100  252  253];
% 251: begin1
% 252: begin2
% 253: end1
% 254: end2
%   1: Standard Stimulus, non Target
%   2: Deviant Stimulus, Target
%  -2: Left Hand Response
%  -4: Right Hand Response
%  -6: Both Hands Response
%   8: marker, stay or go to the right
%   9: marker, stay or go to the right + Standard Stimulus
%   10: marker, stay or go to the right + Deviant Stimulus
%  16: marker, stay or go to the center
%  17: marker, stay or go to the center + Standard Stimulus
%  18: marker, stay or go to the center + Deviant Stimulus
%  32: marker, stay or go to the left
%  33: marker, stay or go to the left + Standard Stimulus
%  34: marker, stay or go to the left + Deviant Stimulus
%  48: curve starts
%  49: curve starts + Standard Stimulus
%  50: curve starts + Deviant Stimulus
%  24: curve ends
%  25: curve ends + Standard Stimulus
%  26: curve ends + Deviant Stimulus
%See:
%   eegfile_readBVmarkers mrkodef_imag_fb01

%Author(s) Claudia Sannelli, October 2008

%mrkdev.indexedByEpochs = {'ishit', 'reac', 'latency', 'iwrong', 'imiss', 'itoofast', 'ileft', 'iright'};

stimDef = {{'S  1', 'S  9', 'S 17', 'S 33', 'S 49', 'S 25'}, {'S  2', 'S  10', 'S 18', 'S 34', 'S 50', 'S 26'}; 'Std', 'Dev'};
respDef = {'R  2', 'R  4', 'R  6'; 'Left', 'Right', 'Both'};
miscDef = {'S 8', 'S 16', 'S 32', 'S252', 'S253', 'S254'; 'goRight', 'goCenter', 'goLeft', 'init', 'timeout', 'timout'};
curveInDef = {'S 48'; 'curveStart'};
curveOutDef = {'S 24'; 'curveEnd'};

mrk_stim = mrk_defineClasses(mrko, stimDef);
mrk_resp = mrk_defineClasses(mrko, respDef);
mrk_misc = mrk_defineClasses(mrko, miscDef);
mrk_curveIn = mrk_defineClasses(mrko, curveInDef);
mrk_curveOut = mrk_defineClasses(mrko, curveOutDef);

fprintf('%d trials rejected: warmup\n', opt.adapt_trials);
idxInit = find(mrk_misc.toe == 252);
if length(idxInit) > 1
    idxInit = idxInit(2);
end
idxTimeout = find(mrk_misc.toe == 253);
if length(idxTimeout) > 1
    idxTimeout = idxTimeout(2);
end
% in VPef (LCT2), strange, end-markers are missed, for this reason check at first
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
mrkstd = mrk_matchStimWithResp(mrk_stim, mrk_resp, 'removevoidclasses', 0, 'missingresponse_policy', 'accept', 'multiresponse_policy', 'first');
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

mrk.iwrong = zeros(1,length(mrk.pos)); 
mrk.ishit(intersect(find(mrk.toe == 1), find(mrk.resp_toe == opt.std_resp_toe))) = 1;
mrk.ishit(intersect(find(mrk.toe == 2), find(mrk.resp_toe ~= opt.std_resp_toe))) = 1;
mrk.iwrong(intersect(find(mrk.toe == 1), find(mrk.resp_toe ~= opt.std_resp_toe))) = 1;
mrk.iwrong(intersect(find(mrk.toe == 2), find(mrk.resp_toe == opt.std_resp_toe))) = 1;
mrk.iwrong(find(mrk.resp_toe == -6)) = 1;

mrk.ileft = ismember(mrk.toe, -2);
mrk.iright = ismember(mrk.toe, -4);
mrk.imiss = ismember(mrk.resp_toe, 'NaN');
mrk = rmfield(mrk,'resp_toe');
mrk.indexedByEpochs = {'reac', 'ishit', 'imiss', 'iwrong', 'ileft', 'iright', 'multiresponse', 'icurve'};
mrk.misc = mrk_misc;
