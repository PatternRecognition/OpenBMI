EEG_RAW_DIR= 'z:\bbciRaw\';

if ~exist(EEG_RAW_DIR, 'dir'),
  fprintf('First connect remote drive z: and rerun %s\n', mfilename);
  fprintf('Windows Explorer: Extras | Netzwerklaufwerk verbinden\n');
  fprintf('  Laufwerk: Z:\n');
  fprintf('  Ordner: \\tubbci2\data\n');
  fprintf('  uncheck ''Verbindung bei Anmeldung wiederherstellen\n');
  fprintf('  click on ''Verbindung unter anderem Benutzernamen herstellen\n');
  fprintf('  and enter your username/passwort and click ''Fertig stellen''.\n');
  return;
end

acq_getDataFolder;
bbci= [];
bbci.setup= 'cspauto';
bbci.train_file= strcat(TODAY_DIR, 'imag_', {'move*','arrow*','audi*'});
bbci.clab= {'not','E*','Fp*','AF*','FAF*','*9','*10','O*','I*','PO7,8'};
bbci.classDef= {1, 2, 3; 'left', 'right', 'foot'};
bbci.classes= 'auto';
bbci.feedback= '1d';
bbci.save_name= strcat(TODAY_DIR, 'bbci_classifier');
bbci.player= 1;
% Unless 'auto' mode does not work robustly:
bbci.setup_opts.usedPat= [1:6];
