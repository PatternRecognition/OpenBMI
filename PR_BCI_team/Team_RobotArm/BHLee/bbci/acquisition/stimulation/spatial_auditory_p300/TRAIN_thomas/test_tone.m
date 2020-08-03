function test_tone(speaker,duration,loudness,opt)

opt.toneDuration=duration;
opt=generateTones(opt);

waves=[opt.cueStream; zeros(2,size(opt.cueStream,2))];

loud=zeros(1,8);
loud(speaker)=loudness;
audio_cue= diag(loud); % loudnes of the cue stimuli


PsychPortAudio('FillBuffer', opt.pahandle, (waves'*audio_cue)');
PsychPortAudio('Start', opt.pahandle);

PsychPortAudio('Stop', opt.pahandle, 1);



 