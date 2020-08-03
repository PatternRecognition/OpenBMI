setup_augcog;

nSubjects= length(augcog);
table= zeros(5, nSubjects);
row_title= {'deviants [\#]', 'standards [\#]', 'reaction time [ms]', ...
            'false negatives [\#]', 'false positives [\#]'};
col_title= {augcog.file};
col_title= apply_cellwise(col_title, 'untex');
fmt_row= '%4.0f';

for tt= {'auditory', 'visual'},
  task= tt{1};
  for cc= {'low', 'high'},
    condition= cc{1};
    table_title= [task ' ' condition];
    for nn= 1:nSubjects,
      classDef= {'D*','S*','T*';'D','S','T'};
      blk= getAugCogBlocks(augcog(nn).file);
      blk= blk_selectBlocks(blk, [condition ' ' task]);
      [cnt, bmrk, Mrk]= readBlocks(augcog(nn).file, blk, classDef);
      table(1,nn)= length(getEventIndices(Mrk, 'D*'));
      table(2,nn)= length(getEventIndices(Mrk, 'S*'));
      mrk= mrk_addResponseLatency(Mrk, {'D*','T*'}, [0 2000]);
      ie= getEventIndices(mrk, 'D*');
      valid= find(~isnan(mrk.latency(ie)));
      table(3,nn)= mean(mrk.latency(ie(valid)));
      table(4,nn)= sum(isnan(mrk.latency(ie)));
      mrk= mrk_addResponseLatency(Mrk, {'S*','T*'}, [0 2000]);
      ie= getEventIndices(mrk, 'S*');
      table(5,nn)= sum(~isnan(mrk.latency(ie)));
    end
    latex_table(['augcog_misc/table_' task '_' condition], table, ...
                'title',table_title, 'fmt_row',fmt_row, ...
                'row_title',row_title, 'col_title',col_title);
  end
  
end
