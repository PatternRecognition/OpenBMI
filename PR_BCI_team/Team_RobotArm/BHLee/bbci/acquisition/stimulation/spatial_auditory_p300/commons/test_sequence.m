
function test_sequence()


for ns =4:8

    sequence = createSequence_THOMAS(200,ns,'repeatable',1,'repetitionBreak',1);
    
    subplot(5,1,ns -3);
    pcolor(reshape(sequence,ns,length(sequence)/ns));
    variance = test_variance(sequence,ns);
    reps = test_repetition_distance(sequence,ns);
    reps
end




function variance = test_variance(sequence,ns)

     sequence = reshape(sequence,ns,length(sequence)/ns);
     
     variance = mean(var(sequence,0,2));

function distance = test_repetition_distance(sequence,ns)
     
     last_occurance = zeros(ns,1);
     min_dist = ones(ns,1)*100;
     for i =1:ns
         last_occurance(i)=find(sequence==i,1,'first');
     end    
     
     for i=ns+1:length(sequence)
         distance = i - last_occurance(sequence(i));
         if distance<min_dist(sequence(i)),
             min_dist(sequence(i))=distance;
         end
         last_occurance(sequence(i))=i;
         
     end
     distance = min(min_dist);
     
     