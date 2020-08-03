%% 506.2 start countdown
%% 683 start 'frisst'
%% 737 start 'keinen'
%% 777 end 'keinen'
%% 974 end 'Gurkensalat'

%% frisst keinen
replay_bb('hexawrite', 7, 'start',683, 'stop', 777);
%% keinen
replay_bb('hexawrite', 7, 'start',737, 'stop', 777);
%% alles
replay_bb('hexawrite', 7, 'start',506.2, 'stop', 974);

%% Option 'speedup', 1 zum durchscannen benutzen!