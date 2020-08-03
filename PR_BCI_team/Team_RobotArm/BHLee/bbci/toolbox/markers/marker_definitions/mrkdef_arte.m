function mrk= mrkdef_arte(Mrk, file, opt)

arte_list= {'stopp', ...
            'Augen links','Augen rechts','Augen oben','Augen unten', ...
            'blinzeln', ...
            'Augen offen & entspannen', 'Augen zu & entspannen', ...
            'beiﬂen', 'Kopf bewegen', ...
            'EMG links', 'EMG rechts'};
arte_list_en= {'stop', ...
               'look left', 'look right', 'look up', 'look down', ...
               'blinking', ...
               'eyes open', 'eyes closed', ...
               'biting', 'head movement', ...
               'hand contraction left', 'hand contraction right'};

Mrk= readMarkerComments(file, opt.fs);
if ~isempty(Mrk.pos),
  mrk= copy_struct(Mrk, 'fs');
  toe= apply_cellwise(Mrk.str, 'strmatch',arte_list);
  invalid= apply_cellwise(toe, 'isempty');
  valid= ~[invalid{:}];
  toe= [toe{valid}];
  mrk.pos= Mrk.pos(valid);
  mrk.y= ind2label(toe);
  mrk.className= arte_list_en(1:max(toe));
else
  warning('marker in arte file not in usual comment format');
  classDef= {'S 10', 'stop';
             'S 11', 'MVC left';
             'S 12', 'MVC right';
             'S 13', 'MVC foot';
             'S  1', 'look left';
             'S  2', 'look right';
             'S  3', 'look up';
             'S  4', 'look down';
             'S  5', 'blinking';
             'S  6', 'eyes open';
             'S  7', 'eyes closed';
             'S  8', 'biting';
             'S  9', 'head movemente'}';
  Mrk= eegfile_readBVmarkers(file, 0);
  mrk= mrk_defineClasses(Mrk, classDef);
  %sum(any(mrk.y,2))
end
mrk= mrk_removeVoidClasses(mrk);
