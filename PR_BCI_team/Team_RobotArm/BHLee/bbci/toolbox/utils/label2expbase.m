function exb=label2expbase(labels)
% label2expbase - generate expbase structure from label (subject_date_paradigm_classes)
%  exb = label2expbase(label)

if ~iscell(labels), labels = {labels}; end

exb = repmat(struct('subject','','date','','paradigm','','appendix','','classes',[]), size(labels));

for i=1:length(labels)
  label = labels{i};
  
  ind = find(label=='_');

  scls = label(ind(end)+1:end);

  switch(scls)
   case 'LR'
    classes = {'left', 'right'};
   case 'LF'
    classes = {'left', 'foot'};
   case 'RF'
    classes = {'right', 'foot'};
   otherwise
    classes = [];
  end

  exb(i).subject = label(1:ind(1)-1);
  exb(i).date    = label(ind(1)+1:ind(4)-1);
  exb(i).paradigm =  label(ind(4)+1:ind(end)-1);
  exb(i).appendix = '';
  exb(i).classes = classes;
end
