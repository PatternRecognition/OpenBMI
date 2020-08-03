#include <assert.h>
#include <dirent.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <allegro.h>
#include <winalleg.h>
#include <windows.h>
#include "lqrender.h"
#include "molecular.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384338
#endif
#define sqr(X) ((X)*(X))
#define rand_double() ((double)rand() / (double)RAND_MAX)
#define distance(X1, Y1, X2, Y2) (sqrt((X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2)))


/***  Display and runtime stuff ***/
BITMAP *screen_buf;
BITMAP *blur_buf;

/* Return a std. normal random number (mean 0, std. dev 1) */
static double
rand_gaussian ()
{
  static int precomputed_next = 0;
  static double next = 0.0;
  double multiplier;
  if (precomputed_next)
    {
      precomputed_next = 0;
      return next;
    }
  else
    {
      double v1, v2, s;
      do
	{
	  v1 = 2 * rand_double () - 1;
	  v2 = 2 * rand_double () - 1;
	  s = v1 * v1 + v2 * v2;
	}
      while (s >= 1 || s == 0);

      multiplier = sqrt (-2 * log (s) / s);
      next = v2 * multiplier;
      precomputed_next = 1;
      return v1 * multiplier;
    }
}

void rotate(double ang, double x, double y, double *ox, double *oy, double cx, double cy)
{
        *ox = (x-cx)*cos(ang) - (y-cy)*sin(ang) + cx;
        *oy = (y-cy)*cos(ang) + (x-cx)*sin(ang) + cy;
        
}

double    sqr_x1 = 45;
double        sqr_y1 = 45;
double        sqr_x2 = 45;
double        sqr_y2 = 525;
double        sqr_x3 = 525;
double        sqr_y3 = 525;        
double        sqr_x4 = 525;
double        sqr_y4 = 45;
double margin = 35;


static double global_ang;


void gravity_wall_field(double x, double y, double dx, double dy, double *foutx, double *fouty, void *usr)
{
    double fx, fy;
    double wall_force = 6;
    fx = sin(global_ang)*1.8;
    fy = cos(global_ang)*1.8;
    double c1x, c2x, c1y, c2y, dax, day, mx, tx, ty, qx;
    
    
    if(x<sqr_x1+margin)
    {

        fx+=wall_force;
        }
    if(x>sqr_x3-margin)
    {

        fx+=-wall_force;
        }
    if(y<sqr_y1+margin)
    {

        fy+=wall_force;
        }
    if(y>sqr_y3-margin)
    {
        fy+=-wall_force;

        }               
        
        
  
     *foutx = fx;
     *fouty = fy;
}

lq_data *the_data;
void stiction_function(double x, double y, double dx, double dy, double *outdx, double *outdy, void *usr)
{
    double speed, height;
    speed = sqrt(dx*dx+dy*dy);
    
    height = get_height(the_data, x, y);
    
    if(height>0.9e7)
    {
        *outdx = dx * 0.90 ;
        *outdy = dy * 0.90;
        
    }
    else
    {
         *outdx = dx * 0.9;
         *outdy = dy * 0.9;
    }
    /*
    if(speed>2)
    {
        
        *outdx = dx * 0.95;
        *outdy = dy * 0.95;
    }
    else
    {
        *outdx = dx * 0.8;
        *outdy = dy * 0.8;
    }
    */
    

}
       
#define DROPLETS 20
void create_liquid_test(void)
{
    int i, j, x, y;
    double fx, fy;
    double ang;
    double x1r, y1r, x2r, y2r,x3r, y3r, x4r, y4r;
    int cx, cy;
    int margin;
    double wall_force = 8;
    lq_table *gauss_table;
    lq_lut *pt[DROPLETS];
    lq_molecule *mol;
    lq_molecule_sim *sim;
    lq_data *data;        
    int edge1, edge2;
    gauss_table = gen_table(256,256);
    set_gaussian_table(gauss_table, 35);
    data = new_data(1024, 1024, sqr_y1, sqr_y3, sqr_x1, sqr_x3);
    sim = new_molecule_sim(-30, 0.08, 0.6, 0.00, -20, 0.9, 2);
    for(i=0;i<DROPLETS;i++)
    {
        pt[i] = new_lut(rand_double()*512,rand_double()*512,gauss_table);
        mol = new_molecule(pt[i]);
        sim_add_molecule(sim, mol);
        data_add_lut(data, pt[i]);        
    }
    the_data = data;
    sim->global_field = gravity_wall_field;
    sim->friction_fn = stiction_function;
          
    while(!key[KEY_ESC])
    {
        clear_to_color(screen_buf, makecol(200,230,250));
      
        
        global_ang = ang = mouse_x/ 120.0;
        //ang = 0;
        cx = SCREEN_W/2-50;
        cy = SCREEN_H/2-50;
                        
        rotate(ang, sqr_x1, sqr_y1, &x1r, &y1r, cx, cy);
        rotate(ang, sqr_x2, sqr_y2, &x2r, &y2r, cx, cy);
        rotate(ang, sqr_x3, sqr_y3, &x3r, &y3r, cx, cy);
        rotate(ang, sqr_x4, sqr_y4, &x4r, &y4r, cx, cy);
        
        line(screen_buf, x1r, y1r, x2r, y2r, makecol(128,128,128));
        line(screen_buf, x2r, y2r, x3r, y3r, makecol(128,128,128));
        line(screen_buf, x3r, y3r, x4r, y4r, makecol(128,128,128));
        line(screen_buf, x4r, y4r, x1r, y1r, makecol(128,128,128));

            
     
        
        
        sim_update_dynamics(sim);

        data_render(data, HEIGHT_SCALE*0.95);  
           
        
        
        for(i=0;i<data->n_list_pts;i++)
        {
            x = data->x_list[i];
            y = data->y_list[i];
             rotate(ang, x, y, &x1r, &y1r, cx, cy);
            putpixel(screen_buf, x1r,y1r, makecol(90,100,120));              
            
            
        }
        
       /* drawing_mode(DRAW_MODE_TRANS, 0, 0, 0);
        set_trans_blender(0,0,0,60);
        for(i=0;i<data->width;i++)
        {
            if(data->n_y_edges[i])
            {
                j =0 ;
                while(j<data->n_y_edges[i])
                {
                    edge1 = data->y_edges[i][0];
                    edge2 = data->y_edges[i][1];
                    vline(screen_buf, i, edge1, edge2, makecol(100,100,230));
                    j+=2;
                }
                    
                   
                    
                }        
        }
        
        solid_mode();*/
        
    for(i=0;i<data->n_list_pts;i++)
        {
            x = data->x_list[i];
            y = data->y_list[i];
    
                         rotate(ang, x, y, &x1r, &y1r, cx, cy);
            putpixel(screen_buf, x1r+2,y1r+2, makecol(255,255,255));              
            
        }
       
    
        
        
        blit(screen_buf, screen, 0,0,0,0, SCREEN_W, SCREEN_H);
        Sleep(5);
    }
    while(!key[KEY_ESC])
    {
        Sleep(100);
    }
}

int
main (int argc, char **argv)
{
  
  
  srand (time (NULL));   
  allegro_init ();
  install_sound (DIGI_AUTODETECT, MIDI_AUTODETECT, NULL);
  install_timer ();
  install_mouse ();
  install_keyboard ();

  
  set_color_depth (32);
  if (set_gfx_mode (GFX_DIRECTX_WIN, 1020, 760, 1020, 760) != 0)
    {
      if (set_gfx_mode (GFX_DIRECTX_WIN, 800, 600, 800, 600) != 0)
	if (set_gfx_mode (GFX_DIRECTX_WIN, 640, 480, 640, 480) != 0)
	  {
	    set_color_depth (24);
	    if (set_gfx_mode (GFX_DIRECTX_WIN, 800, 600, 800, 600) != 0)
	      {
		printf ("Could not create graphics window...\n");
		exit (-1);
	      }


	  }

    }
    screen_buf = create_bitmap(SCREEN_W, SCREEN_H);
    blur_buf = create_bitmap(SCREEN_W, SCREEN_H);
    create_liquid_test();
}
END_OF_MAIN()