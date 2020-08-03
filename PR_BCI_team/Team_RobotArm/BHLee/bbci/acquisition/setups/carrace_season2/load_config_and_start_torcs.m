% Copies the current configuration into the TORCS system configuration
cd(TORCS_DIR)
modify_torcs_settings;
if(strcmp(configFolder,'') == 0)
  absConfigFolder = [TORCS_DIR '\' configFolder '\*'];
  fprintf('Copy files from: %s\n', absConfigFolder);
  copyfile(absConfigFolder,[TORCS_DIR '\']);
end

% Starts up TORCS
fprintf('Starting Torcs.\n');
dos('wtorcs.exe &');
cd(TODAY_DIR)
