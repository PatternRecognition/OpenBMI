function msg = bvr_checkparport()

    msg = 'OK';
    trigger = 2.^[0:7];
    trigger= num2cell(trigger);
      
    try
        bbci_acquire_bv('close');
    catch err
    end
    
    pause(0.2);
    
    params = struct;
	state = bbci_acquire_bv('init',params)

    for tt= 1:length(trigger),
      trig= trigger{tt},
      pause(0.1);
      ppTrigger(trig);
      marker_received= 0;
      tic;
      while toc<1 & ~marker_received,
          
        [data, markertime, markerdescr, state]= bbci_acquire_bv(state);
       
        for mm= 1:length(markerdescr),

            if markerdescr(mm) == trig,              
                marker_received= 1;
            end
        end
      end
    end
                   
    bbci_acquire_bv('close')

    if ~marker_received,
      msg= 'Parport of this computer not connected';
    
      error(msg);
    end;
end
