lowBase = 3000;
highBase = 7500;
toneStart = 440;
toneSteps = 3;
steps = 6; %percent change on boundary
stepsHigh = 7;

for i = 1:length(glo_opt.speakerSelected),
    speaker = glo_opt.speakerSelected(i);
    glo_opt.cueLongStream(speaker,:) = stimutil_filteredNoise(44100, glo_opt.longToneDuration/1000, 3, lowBase, highBase, 3, 3);
    tmpTone = stimutil_generateTone(toneStart*(toneSteps^(1/12))^((i-1)*2), 'duration', glo_opt.longToneDuration, 'pan', [1], 'fs', 44100, 'rampon', 5, 'rampoff', 5);
    toneOverlay = tmpTone(1:length(glo_opt.cueLongStream(speaker,:)),1);
    glo_opt.cueLongStream(speaker,:) = glo_opt.cueLongStream(speaker,:) + (toneOverlay' * 0.15);
    lowBase = lowBase+((steps/100)*lowBase);
    highBase = highBase+((stepsHigh/100)*highBase);
end