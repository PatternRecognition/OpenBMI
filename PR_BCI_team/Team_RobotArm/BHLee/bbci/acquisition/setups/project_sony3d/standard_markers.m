% Adapted from pyff
% by marton, jan 2011

% Markers specifying the start and the end of the run. They should be sent
% when the feedback is started resp. stopped.
marker_run_start = 254;
marker_run_end   = 255;

% Start resp. end of a trial
marker_trial_start = 250;
marker_trial_end   = 251;

% Start resp. end of a countdown
marker_countdown_start = 240;
marker_countdown_end   = 241;

% Onset resp. offset of a fixation marker
marker_fixation_start = 242;
marker_fixation_end   = 243;

% Onset resp. offset of a cue
marker_cue_start = 244;
marker_cue_end   = 245;

% Onset resp. offset of feedback
marker_feedback_start = 246;
marker_feedback_end   = 247;

% Start resp. end of a short break during the run
marker_pause_start = 248;
marker_pause_end   = 249;
