global SINGLE_MODE EEG_TMP_DIR EEG_RAW_DIR REMOTE_RAW_DIR DISPLAY
SINGLE_MODE= 0;
DISPLAY= 'left';
EEG_TMP_DIR = 'C:\data\eeg_temp\';

setup_bbci_bet_unstable;

today_vec= clock;
today_str= sprintf('%02d_%02d_%02d', today_vec(1)-2000, today_vec(2:3));
today_dir= [EEG_TMP_DIR 'tmp_' today_str '\'];
if ~exist(today_dir, 'dir'),
  mkdir_rec(today_dir);
end
if ~exist([today_dir 'log'], 'dir'),
  mkdir_rec([today_dir 'log']);
end
EEG_RAW_DIR= today_dir;
REMOTE_RAW_DIR= today_dir;

general_port_fields.graphic= {'192.168.0.5', 12470};
general_port_fields.control={'192.168.0.6', 12470, 12489};

%bbci.setup= 'cspauto';
bbci.setup= 'csp';
bbci.classDef= {1,      2,       3; ...
                'left', 'right', 'foot'};
%bbci.classes= 'auto';
bbci.classes= {'left','right'};
bbci.feedback= '1d';
bbci.save_name= [today_dir 'classifier'];
bbci.player= 1;

fprintf(['\nRemember to define bbci.train_file before running ' ...
         'bbci_bet_prepare\n']);
