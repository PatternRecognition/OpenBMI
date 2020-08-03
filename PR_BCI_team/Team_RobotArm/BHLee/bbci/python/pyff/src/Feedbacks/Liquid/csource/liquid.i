%module liquid
%{

#include "lqrender.h"
#include "molecular.h"
%}

%include "carrays.i"
%include "cpointer.i"

%array_class(int, intArray);
%array_class(double, doubleArray);
%array_class(lq_section, lqSectionArray);

%array_functions(int, intArr);


%include "lqrender.h"
%include "molecular.h"




%{
  void ForceCallBack(double x, double y, double dx, double dy, double *fx, double *fy, void *clientdata){
   PyObject *func, *arglist;
   PyObject *result;   
   PyObject *self;
 *fx = 0;
 *fy = 0;

   if(!clientdata)
    return;              
   
    
   func = (PyObject *) clientdata;               // Get Python function              
   result = PyEval_CallMethod(func, "getField", "dddd", x, y, dx, dy);     // Call Python
   
   
   if (result) {                                 // If no errors, return double
   
     PyObject *res1, *res2;
     
     res1 = PyTuple_GetItem(result, 0);
     res2 = PyTuple_GetItem(result, 1);
     
     
     *fx = PyFloat_AsDouble(res1);
     *fy = PyFloat_AsDouble(res2);     
     
   }
   else
   {
    printf("Egads! Something went wrong!\n");
    PyErr_Print();
   
   }
   Py_XDECREF(result);
   }
   
     void FrictionCallBack(double x, double y, double dx, double dy, double *fx, double *fy, void *clientdata){
   PyObject *func, *arglist;
   PyObject *result;   
   PyObject *self;
 *fx = 0;
 *fy = 0;

   if(!clientdata)
    return;              
   
    
   func = (PyObject *) clientdata;               // Get Python function              
   result = PyEval_CallMethod(func, "getFriction", "dddd", x, y, dx, dy);     // Call Python
   
   
   if (result) {                                 // If no errors, return double
   
     PyObject *res1, *res2;
     
     res1 = PyTuple_GetItem(result, 0);
     res2 = PyTuple_GetItem(result, 1);
     
     
     *fx = PyFloat_AsDouble(res1);
     *fy = PyFloat_AsDouble(res2);     
     
   }
   else
   {
    printf("Egads! Something went wrong!\n");
    PyErr_Print();
   
   }
   Py_XDECREF(result);
   }

   void set_molecule_field(lq_molecule_sim *sim, PyObject *func)
   {
        sim->global_field_obj = (void *)func;        
        
        Py_XINCREF(func);
        sim->global_field = ForceCallBack;           
   }
   
   void set_molecule_friction(lq_molecule_sim *sim, PyObject *func)
   {
        sim->friction_fn_obj = (void *)func;
        Py_XINCREF(func);
        sim->friction_fn = FrictionCallBack;           
   }
   %}
   
   
   void set_molecule_friction(lq_molecule_sim *sim, PyObject *func);
   void set_molecule_field(lq_molecule_sim *sim, PyObject *func);
   
   