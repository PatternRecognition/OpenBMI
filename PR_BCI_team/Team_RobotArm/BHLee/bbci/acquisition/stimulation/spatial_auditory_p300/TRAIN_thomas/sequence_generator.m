% a target sequence is generated
repetitions=2;

no_durations=7;
nSpeakers=6;
duration_sequence=[];
target_sequence=[];

for i=1:repetitions,
  duration_template=[];
  target_template=[];
  for j=1:no_durations,
    duration_template=[duration_template,ones(1,6)*j];
    target_template=[target_template,1:6];
  end
  while length(duration_template)>0,
  speakers=randperm(nSpeakers);
  for j=1:length(speakers),
    availlable=find(target_template==speakers(j));
    pick=availlable(floor(rand*length(availlable))+1);
    duration_sequence=[duration_sequence , duration_template(pick)];
    target_sequence=[target_sequence , target_template(pick)];
    duration_template(pick)=[];
    target_template(pick)=[];
    
    
    
  end
  
  end


end

save('stimulus_sequence','target_sequence','duration_sequence');
