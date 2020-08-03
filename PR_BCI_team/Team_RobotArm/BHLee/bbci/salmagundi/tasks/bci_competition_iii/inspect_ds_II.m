file_dir= [EEG_IMPORT_DIR 'bci_competition_iii/albany/'];
su= 'A';

file= strcat(file_dir, 'Subject_', su, '_Train');
S= load(file);
