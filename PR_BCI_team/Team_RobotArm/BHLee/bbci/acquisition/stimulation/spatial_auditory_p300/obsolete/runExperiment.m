tic;
output = struct;
output.sequence = [];
target_sequence = [];
Nsingle = 20;
Ndual = 1;

%% Do a round of 'Nsingle' single stimulations
fprintf('Starting with the single stimulus paradigm\n');
if Nsingle < opt.speakerCount;
    target_sequence = randperm(Nsingle);
else
    for ii = 1:Nsingle/opt.speakerCount,
        target_sequence = [target_sequence 1:opt.speakerCount];
    end
    if mod(Nsingle, opt.speakerCount) > 0
        for ii = 1:mod(Nsingle, opt.speakerCount),
            target_sequence = [target_sequence round(rand()*(opt.speakerCount-1))+1];
        end
    end
    target_sequence(:) = target_sequence(randperm(length(target_sequence)));
end

for ii = 1:length(target_sequence),
    opt.targetCue = target_sequence(ii);
    fprintf('Start of trial: %i\n', ii);
    fprintf('Target cue is: %i\n', target_sequence(ii));
    output(ii).sequence = stim_oddballAuditorySpatial(N, opt);
    output(ii).dualStim = opt.dualStim;
    output(ii).target = opt.targetCue;
    fprintf('End of trial: %i\n\n', ii);
    pause(10);
end

time_used_single = toc;

%% Procede to the dual stimulation
%% Do a round of 'Nsingle' single stimulations
pause(1);
fprintf('\n\nProceding with the dual stimulus paradigm\n');
if Ndual < opt.speakerCount;
    target_sequence = randperm(Ndual);
else
    for ii = 1:Ndual/opt.speakerCount,
        target_sequence = [target_sequence 1:opt.speakerCount];
    end
    if mod(Ndual, opt.speakerCount) > 0
        for ii = 1:mod(Ndual, opt.speakerCount),
            target_sequence = [target_sequence round(rand()*(opt.speakerCount-1))+1];
        end
    end
    target_sequence(:) = target_sequence(randperm(length(target_sequence)));
end

opt.dualStim = true;
opt.dualDistance = 1;

for ii = 1:Ndual,
    opt.targetCue = target_sequence(ii);
    fprintf('Start of trial: %i\n', ii+Nsingle);
    fprintf('Target cue is: %i\n', target_sequence(ii));
    output(ii+Nsingle).sequence = stim_oddballAuditorySpatial(N, opt);
    output(ii+Nsingle).dualStim = opt.dualStim;
    output(ii+Nsingle).target = opt.targetCue;
    fprintf('End of trial: %i\n\n', ii+Nsingle);
    pause(10);
end

time_used_dual = toc - time_used_single;
time_single_selection = (time_used_single/Nsingle)-Nsingle*10;
time_dual_selection = (time_used_dual/Ndual)-Ndual*10;

fprintf('\n\nTotal time used for single stimulation: %6.2f\n', time_used_single);
fprintf('Time used per selection with single stimulation: %6.2f\n', time_single_selection);
fprintf('Total time used for single stimulation: %6.2f\n', time_used_dual);
fprintf('Time used per selection with dual stimulation: %6.2f\n', time_dual_selection);

clear ii;