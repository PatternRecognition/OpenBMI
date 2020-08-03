classdef MexFunctionIDs < double
    %MexFunctionID Enumeration that provides ID number of desired function
    %  Copyright 2017 The MathWorks, Inc.

    
    enumeration
        OPEN_LIB (0)
        CLOSE_LIB (1)
        GET_JNT_POS (2)
        GET_JNT_VEL (3)
        GET_JNT_TORQUE (4)
        GET_JNT_TEMP (5)
        GET_EE_POSE (6)
        GET_EE_WRENCH (7)
        MOVE_TO_HOME (8)
        SET_POS_CTRL (9)
        SET_DIRECT_TORQUE_CTRL (10)
        RUN_GRAVITY_CALIB (11)
        INIT_FINGERS (12)
        SEND_JNT_POS (13)
        SEND_JNT_FING_POS (14)
        SEND_JNT_VEL (15)
        SEND_JNT_TORQUE (16)
        SEND_FINGER_POS (17)
        SEND_CART_POS (18)
        SEND_CART_VEL (19)
        GET_DOF (20)
        START_FORCE_CONTROL(21)
        STOP_FORCE_CONTROL(22)
        GET_EE_OFFSET(23)
        SET_EE_OFFSET(24)
        GET_PROTECT_ZONE(25)
        ERR_PROTECT_ZONES(26)
        SET_PROTECT_ZONE(27)
        GET_GLOB_TRAJECTORY (28)
    end
    
    methods
    end
    
end

