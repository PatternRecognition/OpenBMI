function opt = generateTones(opt)
% this function generates the tones for the spatial auditory experiment

% High noise better audible
lowBase = 3000;
highBase = 7500;
toneStart = 440;
toneSteps = 3;
steps = 6; %percent change on boundary
stepsHigh = 7;
opt.cueStream=[];
for i = 1:length(opt.speakerSelected),
    speaker = opt.speakerSelected(i);
    opt.cueStream(speaker,:) = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, lowBase, highBase, 3, 3);
%     opt.cueStream(speaker,:) = stimutil_filteredNoise(44100, opt.toneDuration/1000, 3, 1900, 12000, 3, 3);
    tmpTone = stimutil_generateTone(toneStart*(toneSteps^(1/12))^((i-1)*2), 'duration', opt.toneDuration, 'pan', [1], 'fs', 44100, 'rampon', 3, 'rampoff', 3);
    toneOverlay = tmpTone(1:length(opt.cueStream(speaker,:)),1);
    opt.cueStream(speaker,:) = opt.cueStream(speaker,:) + (toneOverlay' * 0.15);
    lowBase = lowBase+((steps/100)*lowBase);
    highBase = highBase+((stepsHigh/100)*highBase);
end

clear lowBase highBase toneStart toneSteps steps stepsHigh;