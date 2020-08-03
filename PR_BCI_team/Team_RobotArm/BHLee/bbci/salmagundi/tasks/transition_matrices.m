
dir_list= {
'Gabriel_00_09_05/',
'Gabriel_01_07_24/',
'Stefan_01_10_18/',
'Seppel_01_10_19/',
'Pavel_01_11_23/', 
'Gabriel_01_12_12/',
'Roman_01_12_13/',
'Soeren_01_12_17/',
'Guido_02_01_08/',
'Thorsten_02_01_15/',
'Christin_02_10_30/',
'Volker_02_03_05/'};
pace_list= {'2s','1s','0_5s'};

for ff= 1:length(dir_list);

sub_dir= dir_list{ff};
is= min(find(sub_dir=='_'))-1;
subject= sub_dir(1:is);

for pp= 1:length(pace_list),

file= [sub_dir 'selfpaced' pace_list{pp} subject];
if ~exist([EEG_RAW_DIR file '.vmrk'], 'file'), 
  continue; 
end
[dmy,mrk]= loadProcessedEEG(file, '', 'mrk');
idx= getEventPairs(mrk, 3000);
ne= apply_cellwise(idx, 'length');

qq= sum([ne{[2 3]}]) / sum([ne{[1 4]}]);
if qq>1.25,
  mark= '*';
else
  mark= ' ';
end

fprintf('%s%s: ll: %d,  rl: %d,  lr: %d, rr: %d\n', mark, file, ne{:});

end
end
