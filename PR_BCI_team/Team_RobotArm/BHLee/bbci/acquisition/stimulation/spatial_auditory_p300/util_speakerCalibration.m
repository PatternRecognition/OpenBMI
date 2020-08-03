function calibrationMatrix = speaker_calibration(opt)

global BCI_DIR TODAY_DIR

% if length(varargin)>0 & isnumeric(varargin{1}),
%   opt= propertylist2struct('perc_dev',varargin{1}, varargin{2:end});
% else
%   opt= propertylist2struct(varargin{:});
% end
if ~isfield(opt, 'subjectId') || ~isstruct(opt),
    error('Input does not match. Should be <struct> opt');
end

filename = [TODAY_DIR 'Calibration.dat'];

% Establish key mapping: ESCape aborts
ListenChar(2);
numKeysInit = {'1!', '2@', '3#', '4$', '5%', '6^', '7&', '8*'};
detailStep = 0.005;
normalStep = 0.05;

KbName('UnifyKeynames');
esc = KbName('ESCAPE');
volUp = KbName('PageUp');
volDown = KbName('PageDown');
volDetailUp = KbName('UpArrow');
volDetailDown = KbName('DownArrow');
volAllUp = KbName('=+');
volAllDown = KbName('-');
reset = KbName('r');

numKeys = [];
for ii = 1:length(numKeysInit),
    numKeys(1, ii) = KbName(numKeysInit(1, ii));
end

if isempty(opt.speakerSelected),
  adaptChannels = [1:length(numKeysInit)];
else
  adaptChannels = opt.speakerSelected;
end

%% print welcome message
fprintf('Speakers [%s] selected for calibration\n', int2str(sort(opt.speakerSelected)));
fprintf('Press escape to store calibration and exit the routine\n');
fprintf('PageUp and - Down alter volume with large increments, ArrowUp and - Down with small steps\n');
fprintf('+ and - alter the gain off all channels. Button ''r'' resets the levels to the template\n');
fprintf('All buttons are based on the US-International keyboard\n');



resetMatrix = load([BCI_DIR 'acquisition\stimulation\spatial_auditory_p300\CalibrationFiles\calibratedParam.dat'], '-ascii');
try
    calibrationMatrix = load(filename, '-ascii');
    fprintf('Previous calibration for subject found, using it as startingpoint\n');
catch
    calibrationMatrix = resetMatrix;
    warning('No previous calibration file found for subject. Now using standard calibration');
end
% calibrationMatrix = opt.volume;

channelOn = 1;

while 1
    % Check keyboard:
    [isdown dummy, keycode]=KbCheck;
    if isdown
        if keycode(esc)
            break;
        end
        keyindex = find(keycode);
        if length(keyindex) <= 1,
            if ismember(keyindex, numKeys),
                channelOn = find(keyindex == numKeys);
                if ismember(channelOn, adaptChannels),
                  audioStream = zeros(opt.speakerCount, length(opt.cueStream));
                  if size(opt.cueStream, 1) == 1,
                    audioStream(channelOn,:) = opt.cueStream * calibrationMatrix(channelOn, 1);
                  else
                    audioStream(channelOn,:) = opt.cueStream(channelOn,:) * calibrationMatrix(channelOn, 1);
                  end
                  PsychPortAudio('Stop', opt.pahandle);
                  PsychPortAudio('FillBuffer', opt.pahandle, audioStream);
                  PsychPortAudio('Start', opt.pahandle);
                  fprintf('Now calibrating channel: %i\n',channelOn);
                  fprintf('Initial volume on this channel is: %3.3f\n', calibrationMatrix(channelOn, 1));
%                   fprintf('The position should be: %s\n',char(opt.speakerName(channelOn)));
                end
            end
            if keycode(reset)
                calibrationMatrix = resetMatrix;
                fprintf('All gains reset to template calibration\n');
            end
            if keycode(volDetailUp)
                calibrationMatrix(channelOn, 1) = calibrationMatrix(channelOn, 1) + detailStep;
                if calibrationMatrix(channelOn, 1) >= 1
                    calibrationMatrix(channelOn, 1) = 1;
                end
                audioStream = zeros(opt.speakerCount, length(opt.cueStream));
                if size(opt.cueStream, 1) == 1,
                  audioStream(channelOn,:) = opt.cueStream * calibrationMatrix(channelOn, 1);
                else
                  audioStream(channelOn,:) = opt.cueStream(channelOn,:) * calibrationMatrix(channelOn, 1);
                end
                PsychPortAudio('Stop', opt.pahandle);
                PsychPortAudio('FillBuffer', opt.pahandle, audioStream);
                PsychPortAudio('Start', opt.pahandle);
                fprintf('Gain on channel %i is now: %3.3f\n',channelOn,calibrationMatrix(channelOn, 1));
            end
            if keycode(volDetailDown)
                calibrationMatrix(channelOn, 1) = calibrationMatrix(channelOn, 1) - detailStep;
                if calibrationMatrix(channelOn, 1) <= 0
                    calibrationMatrix(channelOn, 1) = 0;
                end
                audioStream = zeros(opt.speakerCount, length(opt.cueStream));
                if size(opt.cueStream, 1) == 1,
                  audioStream(channelOn,:) = opt.cueStream * calibrationMatrix(channelOn, 1);
                else
                  audioStream(channelOn,:) = opt.cueStream(channelOn,:) * calibrationMatrix(channelOn, 1);
                end
                PsychPortAudio('Stop', opt.pahandle);
                PsychPortAudio('FillBuffer', opt.pahandle, audioStream);
                PsychPortAudio('Start', opt.pahandle);
                fprintf('Gain on channel %i is now: %3.3f\n',channelOn,calibrationMatrix(channelOn, 1));
            end
            if keycode(volUp)
                calibrationMatrix(channelOn, 1) = calibrationMatrix(channelOn, 1) + normalStep;
                if calibrationMatrix(channelOn, 1) >= 1
                    calibrationMatrix(channelOn, 1) = 1;
                end
                audioStream = zeros(opt.speakerCount, length(opt.cueStream));
                if size(opt.cueStream, 1) == 1,
                  audioStream(channelOn,:) = opt.cueStream * calibrationMatrix(channelOn, 1);
                else
                  audioStream(channelOn,:) = opt.cueStream(channelOn,:) * calibrationMatrix(channelOn, 1);
                end
                PsychPortAudio('Stop', opt.pahandle);
                PsychPortAudio('FillBuffer', opt.pahandle, audioStream);
                PsychPortAudio('Start', opt.pahandle);
                fprintf('Gain on channel %i is now: %3.3f\n',channelOn,calibrationMatrix(channelOn, 1));
            end
            if keycode(volDown)
                calibrationMatrix(channelOn, 1) = calibrationMatrix(channelOn, 1) - normalStep;
                if calibrationMatrix(channelOn, 1) <= 0
                    calibrationMatrix(channelOn, 1) = 0;
                end
                audioStream = zeros(opt.speakerCount, length(opt.cueStream));
                if size(opt.cueStream, 1) == 1,
                  audioStream(channelOn,:) = opt.cueStream * calibrationMatrix(channelOn, 1);
                else
                  audioStream(channelOn,:) = opt.cueStream(channelOn,:) * calibrationMatrix(channelOn, 1);
                end
                PsychPortAudio('Stop', opt.pahandle);
                PsychPortAudio('FillBuffer', opt.pahandle, audioStream);
                PsychPortAudio('Start', opt.pahandle);
                fprintf('Gain on channel %i is now: %3.3f\n',channelOn,calibrationMatrix(channelOn, 1));
            end
            if keycode(volAllUp)
                calibrationMatrix = calibrationMatrix + normalStep;
                faultyIndex = find(calibrationMatrix > 1);
                calibrationMatrix(faultyIndex) = 1;
                audioStream = zeros(opt.speakerCount, length(opt.cueStream));
                if size(opt.cueStream, 1) == 1,
                  audioStream(channelOn,:) = opt.cueStream * calibrationMatrix(channelOn, 1);
                else
                  audioStream(channelOn,:) = opt.cueStream(channelOn,:) * calibrationMatrix(channelOn, 1);
                end
                PsychPortAudio('Stop', opt.pahandle);
                PsychPortAudio('FillBuffer', opt.pahandle, audioStream);
                PsychPortAudio('Start', opt.pahandle);
                fprintf('Global gain increased with: %3.3f\n',normalStep);
            end
            if keycode(volAllDown)
                calibrationMatrix = calibrationMatrix - normalStep;
                faultyIndex = find(calibrationMatrix < 0);
                calibrationMatrix(faultyIndex) = 1;
                audioStream = zeros(opt.speakerCount, length(opt.cueStream));
                if size(opt.cueStream, 1) == 1,
                  audioStream(channelOn,:) = opt.cueStream * calibrationMatrix(channelOn, 1);
                else
                  audioStream(channelOn,:) = opt.cueStream(opt.cueStream(channelOn,:),:) * calibrationMatrix(channelOn, 1);
                end
                PsychPortAudio('Stop', opt.pahandle);
                PsychPortAudio('FillBuffer', opt.pahandle, audioStream);
                PsychPortAudio('Start', opt.pahandle);
                fprintf('Global gain decreased with: %3.3f\n',normalStep);
            end            
        end
        while KbCheck; end;
    end
    WaitSecs(0.1);
end

% Wait a bit:
WaitSecs(0.1);

%PsychPortAudio('Close', opt.pahandle);
if opt.writeCalibrate
    save(filename, 'calibrationMatrix', '-ascii');
end

% Wait a bit:
WaitSecs(0.1);
ListenChar(0);
return;
