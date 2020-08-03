clear general_port_fields;global general_port_fieldsBV_MACHINE= 'bbcilab';
APPLY_MACHINE= 'bbcilab';FEEDBACK_PLAYER1_MACHINE= 'bbcilab';FEEDBACK_PLAYER2_MACHINE= 'verdandi';general_port_fields= strukt('bvmachine',BV_MACHINE,...
                               'control',{APPLY_MACHINE,12470,12488},...                               'graphic',{FEEDBACK_PLAYER1_MACHINE,12471});
general_port_fields(2)= strukt('bvmachine',BV_MACHINE,...
                               'control',{APPLY_MACHINE,12469,12489},...                               'graphic',{FEEDBACK_PLAYER2_MACHINE,12470});