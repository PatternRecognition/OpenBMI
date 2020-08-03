function markers = bvr_get_marker()
   
    global bvr_state    
    
    markers = '';           
    
   [dmy,dmy,dmy,mt,dmy]= acquire_bv(bvr_state);
   
   for mm= 1:length(mt),      
        markers = [markers ';' mt{mm}];        
   end

end

