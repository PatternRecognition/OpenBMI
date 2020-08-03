function bvr_init_conn()
    
	global bvr_state

    bvr_state = acquire_bv(100, 'localhost');
    bvr_state.reconnect= 1;
    [dmy]= acquire_bv(bvr_state);  %% clear the queue
end
