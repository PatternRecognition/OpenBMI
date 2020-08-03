function freq=get_freq(tone)

%if argument tone is a pitch name (e.g. 'A') output is the corresponding frequency in
%the range C' to B'. if tone is an int between 0 and 11, the number is
%associated with a tone of the chromatic scale (C=0,...,H=11) and the
%corresponding frequency (as above). this is for generating all key -
%probetone combinations for probe_tone_exp.
%
%I.Sturm 11/2007

   switch(tone)
   case {'C',0} 
      freq=523.25;
   case {'C#','Db',1}
      freq=554.37;
   case {'D',2}
      freq=587.33;
   case {'D#','Eb',3}
      freq=622.25;
   case {'E',4}
      freq=659.26;
   case {'F',5}
      freq=698.46;
   case {'F#','Gb',6}
      freq=739.99;
   case {'G',7}
      freq=783.99;
   case {'G#','Ab',8}
      freq=830.61;
   case {'A',9}
      freq=880;
   case {'A#','Bb',10}
      freq=932.33;
   case {'B',11}
      freq=987.77;
      
      
   end
   
   freq=freq*0.5;