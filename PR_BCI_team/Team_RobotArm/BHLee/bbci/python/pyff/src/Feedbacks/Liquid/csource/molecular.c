#include <math.h>
#include "molecular.h"

void default_friction_fn(double x, double y, double dx, double dy, double *outdx, double *outdy, void *usr)
{
    *outdx = dx * 0.9;
    *outdy = dy * 0.9;
   
}

//creates a new molecule, and puts it in the global list
lq_molecule_sim *new_molecule_sim(double rep_scale, double rep_size, double dt, double temperature, double att_scale, double att_size, double min_distance)
{
    lq_molecule_sim *sim;    
    sim = calloc(sizeof(*sim), 1);
    sim->temperature = temperature;
    sim->global_field = NULL;
    sim->rep_scale = rep_scale;
    sim->min_distance = min_distance;
    sim->rep_size =rep_size;    
    sim->att_scale = att_scale;
    sim->att_size = att_size;
    sim->friction_fn = default_friction_fn;
    sim->dt = dt;
    sim->molecule_list = NULL;
    return sim;
}


//creates a new molecule, and puts it in the global list
lq_molecule *new_molecule(lq_lut *lut)
{
    lq_molecule *mol;    
    mol = calloc(sizeof(*mol), 1);
    mol->lut = lut;
    mol->x = lut->x_offset;
    mol->y = lut->y_offset;
    mol->dx = 0;
    mol->dy = 0;    
    mol->ddx = 0;
    mol->ddy = 0;    
    mol->energy = 0;
    mol->fmult = 1;
    return mol;
}

void sim_add_molecule(lq_molecule_sim *sim, lq_molecule *mol)
{
    mol->next = sim->molecule_list;
    sim->molecule_list = mol;
}


void sim_update_dynamics(lq_molecule_sim *sim)
{
    lq_molecule *test_mol;
    test_mol = sim->molecule_list;
    while(test_mol!=NULL)
    {
        molecule_update_dynamics(test_mol, sim);
        test_mol = test_mol->next;    
    }  
}

void molecule_update_dynamics(lq_molecule *mol, lq_molecule_sim *sim)
{
    double rep_x, rep_y, att_x, att_y, fx, fy, dfx, dfy;
    double force_x, force_y;
    double d, drep, datt;
    double diff_d;
    double dt = sim->dt;
    int d1, d2;
    lq_molecule *test_mol;
    
     //compute replusive + attractive force
    rep_x = rep_y = 0;
    att_x = att_y = 0;
    test_mol = sim->molecule_list;    
    force_x = force_y = 0;
    
    
    mol->energy = 0;
    while(test_mol!=NULL)
    {        
        if(test_mol->lut->blob_class == mol->lut->blob_class)
        {
            d1 =  (test_mol->lut->x_offset - mol->lut->x_offset);
            d2 =  (test_mol->lut->y_offset - mol->lut->y_offset);
            
            if(fabs(d1) < mol->lut->table->falloff_distance && 
               fabs(d2) < mol->lut->table->falloff_distance && 
               fabs(d1)>0 && fabs(d2)>0)
            {
                
                 d = sqrt(d1*d1 + d2*d2)+sim->min_distance;
                
                drep = (d)*(d)*(d);
                datt = ((d)*(d)*(d)) * sim->att_size;
                
         
                drep *= sim->rep_size;
                
                
                //impacts
                //diff_d = sqrt((mol->dx-test_mol->dx)*(mol->dx-test_mol->dx)+(mol->dy-test_mol->dy)*(mol->dy-test_mol->dy));
                
                //mol->energy += diff_d;
                
                rep_x += (d1/(drep));
                rep_y += (d2/(drep));
                att_x -= (d1/(datt));
                att_y -= (d2/(datt));                
                
            }
           }
            test_mol = test_mol->next;
    }    
    force_x += rep_x * sim->rep_scale + att_x * sim->att_scale;
    force_y += rep_y * sim->rep_scale + att_y * sim->att_scale;
    
    //potential field
    if(sim->global_field!=NULL)
    {
        sim->global_field(mol->x, mol->y, mol->dx, mol->dy, &fx, &fy, sim->global_field_obj);
        force_x += fx;
        force_y += fy;    
    }
         
    
   
    
    mol->dx += force_x * dt;
    mol->dy += force_y * dt;
    
     if(sim->friction_fn!=NULL)
    {
        sim->friction_fn(mol->x, mol->y, mol->dx, mol->dy, &dfx, &dfy, sim->friction_fn_obj);
        
        mol->energy = ((mol->dx - dfx) * (mol->dx-dfx) + (mol->dy-dfy)*(mol->dy-dfy));
        
        mol->dx = dfx * mol->fmult;
        mol->dy = dfy * mol->fmult;            
    }
    
    //temperature jitter
    if(rand_double()<0.01)
    {
        mol->dx += (rand_double() * (sim->temperature))-sim->temperature/2;
        mol->dy += (rand_double() * (sim->temperature))-sim->temperature/2;
    }
    
    mol->x += mol->dx * dt;
    mol->y += mol->dy * dt;
    
 
    //move actual particle
    mol->lut->x_offset = mol->x;
    mol->lut->y_offset = mol->y;    
       
}


   
   