% Eyes-open/eyes-closed after the main experiment

addpath([BCI_DIR 'acquisition/setups/season10']);

if isempty(TODAY_DIR)
  today_vec = clock;
  TODAY_DIR = [EEG_RAW_DIR VP_CODE sprintf('_%02d_%02d_%02d/', today_vec(1)-2000, today_vec(2:3))];
  mkdir_rec(TODAY_DIR)
end
fprintf('\n\nRelax recording.\n');
[seq, wav, opt] = setup_season10_relax;
fprintf('Press <RETURN> to start RELAX measurement. \n'); pause;
fprintf('Ok, starting...\n');
stim_artifactMeasurement(seq, wav, opt);