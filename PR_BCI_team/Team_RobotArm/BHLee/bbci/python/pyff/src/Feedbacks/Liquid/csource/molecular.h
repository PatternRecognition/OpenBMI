#include "lqrender.h"
typedef void (*potential_field) (double , double , double , double , double *, double *, void *);
typedef void (*friction_function) (double , double , double , double , double *, double *, void *);

//A  molecule
typedef struct lq_molecule {
lq_lut *lut;
double x, y;
double dx, dy;
double ddx, ddy;
double energy;
double fmult;
struct lq_molecule *next;
} lq_molecule;

// a set of molecules
typedef struct lq_molecule_sim {
lq_molecule *molecule_list;
double temperature;
double dt;
double rep_scale, rep_size;
double att_scale, att_size, min_distance;
potential_field global_field;
friction_function friction_fn;

void *global_field_obj;
void *friction_fn_obj;
} lq_molecule_sim;

lq_molecule *new_molecule(lq_lut *lut);
lq_molecule_sim *new_molecule_sim(double rep_scale, double rep_size, double dt, double temperature, double att_scale, double att_size, double min_distance);
void sim_add_molecule(lq_molecule_sim *sim, lq_molecule *mol);
void sim_update_dynamics(lq_molecule_sim *sim);
void molecule_update_dynamics(lq_molecule *mol, lq_molecule_sim *sim);