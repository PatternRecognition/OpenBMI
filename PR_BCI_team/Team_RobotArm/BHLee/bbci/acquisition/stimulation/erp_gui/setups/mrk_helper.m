function mrk = mrk_helper(mrko),

mrk = mrkodef_Spatialoddball(mrko, struct('individual', 0));
mrk = mrk_selectClasses(mrk, [2 1]);
mrk.className = {'Target', 'Non-target'};
mrk = mrk_addInfo_P300design(mrk, 6, 15);
mrk = mrk_addIndexedField(mrk,'stimulus');
mrk.stimulus = mrk.toe;
mrk.stimulus(find(mrk.stimulus > 10)) =mrk.stimulus(find(mrk.stimulus > 10))-10;