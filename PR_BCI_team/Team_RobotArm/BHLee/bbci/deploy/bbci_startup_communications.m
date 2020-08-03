function bbci_startup_communications( parallel_port_addr, pyff_port, path )

    global IO_ADDR
    global IO_LIB
    global general_port_fields
    
    general_port_fields = struct(   'bvmachine','127.0.0.1',...
                                    'control',{{'127.0.0.1',12471,12487}},...
                                    'graphic',{{'',12487}});
                        
    general_port_fields.feedback_receiver= 'pyff';
    general_port_fields.bvmachine = '127.0.0.1';           
    
    IO_ADDR = str2num(parallel_port_addr);
    IO_LIB = [path '\inpout32.dll'];    

    %send_xmlcmd_udp('init', general_port_fields.bvmachine, str2num(pyff_port));       
    send_udp_xml('init', general_port_fields.bvmachine, str2num(pyff_port));       
end

