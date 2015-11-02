[fid, message]=fopen(fullfile(BMI.EEG_RAW_DIR, [file '.vhdr']));
a = ismember(cell_order_all(count_run,:), target_ind);

randi(length(spell_char2 ),1)